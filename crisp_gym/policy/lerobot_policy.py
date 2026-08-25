"""Interface for a Policy interacting in CRISP."""

import json
import logging
import multiprocessing
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import LeRobotDatasetMetadata, get_policy_class
from typing_extensions import override

from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv
from crisp_gym.policy.policy import Action, Observation, Policy, register_policy
from crisp_gym.util.fiper_utils import next_fiper_episode_index
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

try:
    from lerobot.policies.factory import make_pre_post_processors
    USE_LEROBOT_PROCESSORS = True
    logging.info("Found lerobot pre/post processor support.")
except ImportError:
    USE_LEROBOT_PROCESSORS = False
    logging.warning("No lerobot pre/post processor support found.")


logger = logging.getLogger(__name__)


@register_policy("lerobot_policy")
class LerobotPolicy(Policy):
    """A policy implementation that wraps a LeRobot policy for use in CRISP environments.

    This class runs LeRobot policy inference in a separate process and communicates with the
    environment to generate actions based on observations. It is intended for direct use in
    CRISP-based manipulation environments.
    """

    def __init__(
        self,
        pretrained_path: str,
        env: ManipulatorBaseEnv,
        overrides: dict | None = None,
        task: str | None = None,
        fiper_recorder_config: Any | None = None,
        fiper_output_dir: Path | str | None = None,
    ):
        """Initialize the policy.

        Args:
            pretrained_path (str): Path to the pretrained policy model.
            env (ManipulatorBaseEnv): The environment in which the policy will be applied.
            overrides (dict | None): Optional overrides for the policy configuration.
            task (str | None): Task description for language-conditioned policies.
            fiper_recorder_config (Any | None): Optional FIPER rollout recorder config.
            fiper_output_dir (Path | str | None): Optional root directory for FIPER rollouts.
        """
        self.env = env
        self.overrides = overrides if overrides is not None else {}
        self.task = task
        self.fiper_recording_enabled = fiper_recorder_config is not None

        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()

        # Extract env data before spawning (env may not be picklable — ROS2 handles)
        observation_space = env.observation_space
        env_metadata = env.get_metadata()

        self.inf_proc = ctx.Process(
            target=inference_worker,
            kwargs={
                "conn": self.child_conn,
                "pretrained_path": pretrained_path,
                "observation_space": observation_space,
                "env_metadata": env_metadata,
                "overrides": self.overrides,
                "task": self.task,
                "fiper_recorder_config": fiper_recorder_config,
                "fiper_output_dir": Path(fiper_output_dir) if fiper_output_dir is not None else None,
            },
            daemon=True,
        )
        self.inf_proc.start()

    @override
    def make_data_fn(self) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""

        def _fn() -> tuple:
            """Function to apply the policy in the environment.

            This function observes the current state of the environment, sends the observation
            to the inference worker, receives the action, and steps the environment.

            Returns:
                tuple: A tuple containing the observation from the environment and the action taken.
            """
            logger.debug("Requesting action from policy...")
            obs_raw: Observation = self.env.get_obs()

            obs_raw["observation.state"] = concatenate_state_features(obs_raw)
            obs_raw["observation.state"][2] -= 0.019     # DEBUG: compensate for force-torque sensor

            if self.task is not None:
                obs_raw["task"] = self.task

            self.parent_conn.send(obs_raw)
            action: Action = self.parent_conn.recv().squeeze(0).to("cpu").numpy()
            action[3:5] *= 0.1  # DEBUG: no orientation command
            # if obs_raw["observation.state"][2] <= 0.095:     # DEBUG: avoid driving into the table
            # if obs_raw["observation.state"][2] <= 0.085:        # For task 2
            #     action[2] = max(action[2], 0.0)
            # if action[6] <= 0.2 or (obs_raw["observation.state"][6] >= 0.6 and action[6] <= 0.4):              # Tasks 0 and 1
            # if action[6] <= 0.2:
            #     action[6] = 0.01
            logger.debug(f"Action: {action}")

            try:
                self.env.step(action, block=False)
                # pass
            except Exception as e:
                logger.exception(f"Error during environment step: {e}")

            return obs_raw, action

        return _fn

    def set_task(self, task: str):
        """Update the language instruction for subsequent inference steps."""
        logger.info(f"[Policy] Task changed to: '{task}'")
        self.task = task

    @override
    def reset(self):
        """Reset the policy state."""
        self.parent_conn.send("reset")

    @override
    def shutdown(self):
        """Shutdown the policy and release resources."""
        self.parent_conn.send(None)
        self.inf_proc.join()

    @override
    def save_fiper_rollout(self, metadata: dict[str, Any]) -> None:
        """Request that the inference worker persist buffered FIPER rollout data."""
        if not self.fiper_recording_enabled:
            raise RuntimeError("FIPER rollout recording is not enabled for this policy.")
        self.parent_conn.send({"type": "SAVE_FIPER", "metadata": metadata})

    @override
    def delete_fiper_rollout(self) -> None:
        """Request that the inference worker discard buffered FIPER rollout data."""
        if not self.fiper_recording_enabled:
            return
        self.parent_conn.send({"type": "DELETE_FIPER"})


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    observation_space: Any,
    env_metadata: dict,
    overrides: dict | None = None,
    task: str | None = None,
    fiper_recorder_config: Any | None = None,
    fiper_output_dir: Path | None = None,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        observation_space: The environment's observation space (pre-extracted for spawn compatibility).
        env_metadata (dict): The environment metadata (pre-extracted for spawn compatibility).
        overrides (dict | None): Optional overrides for the policy configuration.
        task (str | None): Task description for language-conditioned policies.
        fiper_recorder_config (Any | None): Optional FIPER rollout recorder config.
        fiper_output_dir (Path | None): Optional root directory for FIPER rollouts.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except ImportError:
        logger.warning(
            "[Inference] Could not import third-party plugins for LeRobot. Continuing without them."
        )
    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")

        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)

        _check_dataset_metadata(train_config, env_metadata, logger)

        logger.info("[Inference] Loaded training config.")

        logger.debug(f"[Inference] Train config: {train_config}")

        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)

        model_config = policy.config
        for override_key, override_value in (overrides or {}).items():
            logger.warning(
                f"[Inference] Overriding policy config: {override_key} = {getattr(model_config, override_key)} -> {override_value}"
            )
            setattr(model_config, override_key, override_value)

        logger.info(f"[Inference] num_steps = {model_config.num_steps}")

        # logger.info(
        #     f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        # )
        policy.reset()
        policy.to(device).eval()

        if USE_LEROBOT_PROCESSORS:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=pretrained_path,
            )

            # logger.info(f"[Inference] Normalization stats loaded from: {pretrained_path}")
            # for name, pipeline in [("preprocessor", preprocessor), ("postprocessor", postprocessor)]:
            #     for step in pipeline.steps:
            #         if hasattr(step, "stats") and step.stats:
            #             logger.info(f"[Inference] {name} step {step.__class__.__name__} stats: {step.stats}")

        warmup_obs_raw = observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        if task is not None:
            warmup_obs_raw["task"] = task
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)
        if USE_LEROBOT_PROCESSORS:
            warmup_obs = preprocessor(warmup_obs)
            warmup_obs.pop("action", None)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Inference] Warm-up complete")

        fiper_recorder_host = None
        if fiper_recorder_config is not None:
            fiper_recorder_host = _get_fiper_recorder_host(policy)
            if fiper_recorder_host is None:
                logger.warning(
                    "[Inference] FIPER config was provided, but this policy does not expose "
                    "init_fiper_rollout_recorder()."
                )
            else:
                fiper_recorder_host.init_fiper_rollout_recorder(config=fiper_recorder_config)
                logger.info("[Inference] Attached FIPER rollout recorder.")

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                if USE_LEROBOT_PROCESSORS:
                    preprocessor.reset()
                    postprocessor.reset()
                continue
            if isinstance(obs_raw, dict) and obs_raw.get("type") == "SAVE_FIPER":
                _save_fiper_rollout(
                    recorder_host=fiper_recorder_host,
                    output_dir=fiper_output_dir,
                    metadata=obs_raw.get("metadata", {}),
                    model_config=model_config,
                    recorder_config=fiper_recorder_config,
                    logger=logger,
                )
                continue
            if isinstance(obs_raw, dict) and obs_raw.get("type") == "DELETE_FIPER":
                _delete_fiper_rollout(recorder_host=fiper_recorder_host, logger=logger)
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                if USE_LEROBOT_PROCESSORS:
                    obs = preprocessor(obs)
                    obs.pop("action", None)

                action = policy.select_action(obs)

                if USE_LEROBOT_PROCESSORS:
                    action = postprocessor(action)

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")


def _get_fiper_recorder_host(policy: Any) -> Any | None:
    """Return the object that owns the FIPER rollout recorder API."""
    if hasattr(policy, "init_fiper_rollout_recorder"):
        return policy

    if hasattr(policy, "get_base_model"):
        base_model = policy.get_base_model()
        if hasattr(base_model, "init_fiper_rollout_recorder"):
            return base_model

    return None


def _save_fiper_rollout(
    recorder_host: Any | None,
    output_dir: Path | None,
    metadata: dict[str, Any],
    model_config: Any,
    recorder_config: Any | None,
    logger: logging.Logger,
) -> None:
    """Persist buffered FIPER rollout data from the inference worker."""
    if recorder_host is None:
        logger.warning("[Inference] SAVE_FIPER received but no rollout recorder is attached.")
        return

    recorder = getattr(recorder_host, "fiper_rollout_recorder", None)
    if recorder is None:
        logger.warning("[Inference] SAVE_FIPER received but no rollout recorder is attached.")
        return

    if output_dir is None:
        logger.warning("[Inference] SAVE_FIPER received but no output directory is configured.")
        return

    ep_metadata = metadata.copy()
    rollout_type = ep_metadata.get("rollout_type")
    if rollout_type is None:
        logger.warning("[Inference] SAVE_FIPER metadata is missing rollout_type.")
        return

    rollout_dir = output_dir / rollout_type
    ep_metadata.update(
        episode=next_fiper_episode_index(output_dir=rollout_dir),
        action_prediction_horizon=_first_config_value(
            model_config,
            "chunk_size",
            "horizon",
            "n_action_steps",
        ),
        action_execution_horizon=_first_config_value(model_config, "n_action_steps"),
    )
    action_batch_size = getattr(recorder_config, "num_uncertainty_sequences", None)
    if action_batch_size is not None:
        ep_metadata["action_batch_size"] = action_batch_size

    recorder.save_data(output_dir=rollout_dir, episode_metadata=ep_metadata)
    logger.info(
        f"[Inference] Saved FIPER rollout data to {rollout_dir} "
        f"(episode {ep_metadata.get('episode')})."
    )


def _delete_fiper_rollout(recorder_host: Any | None, logger: logging.Logger) -> None:
    """Discard buffered FIPER rollout data from the inference worker."""
    if recorder_host is None:
        logger.warning("[Inference] DELETE_FIPER received but no rollout recorder is attached.")
        return

    recorder = getattr(recorder_host, "fiper_rollout_recorder", None)
    if recorder is None:
        logger.warning("[Inference] DELETE_FIPER received but no rollout recorder is attached.")
        return

    recorder.reset()
    logger.info("[Inference] Deleted FIPER rollout buffer.")


def _first_config_value(config: Any, *names: str) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _check_dataset_metadata(
    train_config: TrainPipelineConfig,
    env_metadata: dict,
    logger: logging.Logger,
    keys_to_skip: list[str] | None = None,
):
    """Check if the dataset metadata matches the environment configuration.

    Args:
        train_config (TrainPipelineConfig): The training pipeline configuration.
        env_metadata (dict): The environment metadata dict to compare against.
        logger (logging.Logger): Logger for logging information.
        keys_to_skip (list[str] | None): List of metadata keys to skip during comparison.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    def _warn_if_not_equal(key: str, env_val: Any, policy_val: Any):
        if env_val != policy_val:
            logger.warning(
                f"[Inference] Mismatch in metadata for key '{key}': "
                f"env has '{env_val}', policy has '{policy_val}'."
            )

    def _warn_if_missing(key: str):
        logger.warning(f"[Inference] Key '{key}' not found in environment metadata.")

    try:
        metadata = LeRobotDatasetMetadata(repo_id=train_config.dataset.repo_id)
        logger.debug(f"[Inference] Loaded dataset metadata: {metadata}")

        path_to_metadata = Path(metadata.root / "meta" / "crisp_meta.json")
        if path_to_metadata.exists():
            logger.info(
                "[Inference] Found crisp_meta.json in dataset, comparing environment and policy configs..."
            )
            with open(path_to_metadata, "r") as f:
                dataset_metadata = json.load(f)
            for key, value in dataset_metadata.items():
                if key in keys_to_skip:
                    continue
                if isinstance(value, dict):
                    if key not in env_metadata:
                        _warn_if_missing(key)
                        continue
                    for subkey, subvalue in value.items():
                        if subkey not in env_metadata[key]:
                            _warn_if_missing(f"{key}.{subkey}")
                            continue
                        _warn_if_not_equal(
                            f"{key}.{subkey}",
                            env_metadata[key].get(subkey),
                            subvalue,
                        )
                else:
                    if key not in env_metadata:
                        _warn_if_missing(key)
                    _warn_if_not_equal(key, env_metadata.get(key), value)

    except Exception as e:
        logger.warning(f"[Inference] Could not load dataset metadata: {e}")
        logger.info("[Inference] Skipping metadata comparison.")
