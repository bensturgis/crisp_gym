"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.fiper_data_recorder.configuration_fiper_data_recorder import FiperDataRecorderConfig
from lerobot.policies.factory import (
    get_policy_class,
    make_pre_post_processors,
)
from lerobot.uncertainty.uncertainty_scoring.scorer_artifacts import (
    build_scorer_artifacts_for_fiper_recorder,
)

from crisp_gym.util.control_type import ControlType
from crisp_gym.util.lerobot_features import numpy_obs_to_torch

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from crisp_gym.manipulator_env import ManipulatorBaseEnv
    from crisp_gym.teleop.teleop_robot import TeleopRobot


def make_teleop_fn(env: ManipulatorBaseEnv, leader: TeleopRobot) -> Callable:
    """Create a teleoperation function for the leader robot.

    This function returns a Callable that can be used to control the leader robot
    in a teleoperation manner. It computes the action based on the difference
    between the current and previous end-effector pose or joint values, and
    updates the gripper value based on the leader gripper's value.

    Args:
        env (ManipulatorBaseEnv): The environment in which the leader robot operates.
        leader (TeleopRobot): The teleoperation leader robot instance.

    Returns:
        Callable: A function that, when called, performs a step in the environment
        and returns the observation and action taken.
    """
    prev_pose = leader.robot.end_effector_pose
    prev_joint = leader.robot.joint_values
    first_step = True

    def _fn() -> tuple:
        """Teleoperation function to be called in each step.

        This function computes the action based on the current end-effector pose
        or joint values of the leader robot, updates the gripper value, and steps
        the environment.

        Returns:
            tuple: A tuple containing the observation from the environment and the action taken.
        """
        nonlocal prev_pose, prev_joint, first_step
        if first_step:
            first_step = False
            prev_pose = leader.robot.end_effector_pose
            prev_joint = leader.robot.joint_values
            return None, None

        pose = leader.robot.end_effector_pose
        joint = leader.robot.joint_values
        action_pose = pose - prev_pose
        action_joint = joint - prev_joint
        prev_pose = pose
        prev_joint = joint

        if leader.gripper is None:
            gripper = 0.0
        elif env.gripper is None:
            gripper = leader.gripper.value + np.clip(
                leader.gripper.value - leader.gripper.value,
                -leader.gripper.config.max_delta,
                leader.gripper.config.max_delta,
            )
        else:
            gripper = env.gripper.value + np.clip(
                leader.gripper.value - env.gripper.value,
                -env.gripper.config.max_delta,
                env.gripper.config.max_delta,
            )

        action = None
        if env.ctrl_type is ControlType.CARTESIAN:
            action = np.concatenate(
                [
                    list(action_pose.position) + list(action_pose.orientation.as_euler("xyz")),
                    [gripper],
                ]
            )
        elif env.ctrl_type is ControlType.JOINT:
            action = np.concatenate(
                [
                    action_joint,
                    [gripper],
                ]
            )
        else:
            raise ValueError(
                f"Unsupported control type: {env.ctrl_type}. "
                "Supported types are 'cartesian' and 'joint'."
            )

        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
    steps: int | None,
    inpainting: bool,
    replan_time: int,
    fiper_recorder_config: FiperDataRecorderConfig | None = None,
):
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
        steps (int): How many actions are executed from the prediction
        inpainting (bool): Wether to use inpainting in the prediction of a new chunk or not 
        replan_time (int): After how many steps to start predicting a new action chunk 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)    
    if steps is not None:
        # Check if the number of steps make sense 
        horizon=policy_config.horizon
        if steps >= horizon: 
            raise ValueError(
            f"The policy steps={steps} must be smaller than the horizon={horizon}."
            "Please modify your cli."
        )
        policy_config.n_action_steps = int(steps)

    if inpainting is True:
        policy_config.inpainting_lengh = max(0, int(policy_config.n_action_steps) - int(replan_time))
    
    if Path(pretrained_path).is_dir():
        config_path = Path(pretrained_path) / "train_config.json"
    elif Path(pretrained_path).is_file():
        config_path = Path(pretrained_path)
    with open(config_path, "r") as f:
        policy_type = json.load(f)["policy"]["type"] 
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(pretrained_path, config=policy_config)

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
    )

    policy.reset()
    policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=pretrained_path,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    warmup_obs_raw = env.observation_space.sample()
    warmup_obs = numpy_obs_to_torch(warmup_obs_raw)

    with torch.inference_mode():
        _ = policy.select_action(warmup_obs)
        torch.cuda.synchronize()

    logging.info("[Inference] Warm-up complete")

    if fiper_recorder_config is not None:
        scorer_artifacts = build_scorer_artifacts_for_fiper_recorder(
            fiper_data_recorder_cfg=fiper_recorder_config,
            policy_cfg=policy_config,
            env_cfg=cfg.env,
            dataset_cfg=cfg.dataset,
            policy=policy,
            preprocessor=preprocessor,
        )

    logging.info("Ready to recive information")
    while True:
        # Check if messages have been received correctly
        msg = conn.recv()
        if msg is None:
            break
        if msg == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            continue
        if not (isinstance(msg, dict) and msg.get("type") == "OBS_SEQ"):
            logging.warning(f"[Inference] Unknown message: {type(msg)}")
            continue
        
        # We are receiving a list of dictonaries with the last observations 
        obs_seq = msg["obs_seq"]

        # Make the policy predict an action chunk for the current obeservation.
        # Therefore we follow the implementation on the Lerobot side for select_action() which calls predict_action_chunk()
        with torch.inference_mode():
            for i in range(policy.config.n_obs_steps):
                obs = numpy_obs_to_torch(obs=obs_seq[i], env=env)
                obs = preprocessor(obs)

            # Now get a fresh chunk
            actions = policy.predict_action_chunk(obs)  
            actions = actions.transpose(0, 1)[: policy.config.n_action_steps]

        actions = np.stack(
            [postprocessor(a).squeeze(0).to("cpu").numpy() for a in actions],
            axis=0
        )

        logging.debug(f"[Inference] Computed action chunk with shape {tuple(actions.shape)}")
        conn.send(actions)

    conn.close()
    logging.info("[Inference] Worker shutting down")

