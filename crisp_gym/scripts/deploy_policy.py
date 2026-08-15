"""Script showcasing how to record data in Lerobot Format."""

import argparse
import datetime
import logging
import threading
from pathlib import Path

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.policy import make_policy
from crisp_gym.policy.policy import list_policy_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.fiper_utils import collect_fiper_metadata, load_fiper_recorder_config
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging


def main():
    """Deploy a pretrained policy and record deployment data in Lerobot Format."""
    parser = argparse.ArgumentParser(
        description="Deploy a pretrained policy and record data in Lerobot Format"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Type of robot being used.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for recording",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume recording of an already existing dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to push the dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--recording-manager-type",
        type=str,
        default="keyboard",
        help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
    )
    parser.add_argument(
        "--joint-control",
        action="store_true",
        help="Whether to use joint control for the robot.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logger level.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the pretrained model (if not provided, a prompt will ask you to select one from 'outputs/train')",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="left_robot_env",
        help="Configuration name for the follower robot. You can define your own configurations, please check https://utiasdsl.github.io/crisp_controllers/misc/create_own_config/.",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default=None,
        help="Path to a custom policy configuration file (YAML). If not provided, the default configuration for the selected policy will be used.",
    )
    parser.add_argument(
        "--env-namespace",
        type=str,
        default="left",
        help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Whether to evaluate the performance of the model after each episode.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["put the lego into the drawer"],
        help="Task descriptions for language-conditioned policies. "
        "Multiple tasks can be provided for timed switching "
        "(e.g. --tasks 'put the lego into the drawer' 'close the drawer'). "
        "Known dataset tasks include "
        "'put the bowl on the plate' "
        "for continuallearning/real_0_put_bowl_filtered_consolidated, "
        "'stack the orange bowl on the other bowls' "
        "for continuallearning/real_1_stack_bowls_filtered_consolidated, and "
        "'put the lego into the drawer' "
        "for continuallearning/real_4_put_lego_into_drawer_filtered_consolidated.",
    )
    parser.add_argument(
        "--switch-at",
        type=float,
        nargs="+",
        default=None,
        help="Times (in seconds) at which to switch to the next task. "
        "Number of values must be len(tasks) - 1. "
        "(e.g. --switch-at 2.0 means switch to the 2nd task at t=2s).",
    )
    parser.add_argument(
        "--episode-length",
        "--episode_length",
        dest="episode_length",
        type=int,
        default=None,
        help="Auto-stop an episode after this many environment steps.",
    )
    parser.add_argument(
        "--fiper-config",
        type=str,
        default=None,
        help="Path to a FIPER data recorder config file.",
    )
    parser.add_argument(
        "--fiper-output-dir",
        type=str,
        default=None,
        help="Directory containing FIPER rollouts/calibration and rollouts/test files.",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    # Validate --switch-at against --tasks
    if args.switch_at is not None:
        if len(args.switch_at) != len(args.tasks) - 1:
            parser.error(
                f"--switch-at requires exactly {len(args.tasks) - 1} value(s) "
                f"for {len(args.tasks)} tasks, got {len(args.switch_at)}"
            )
    elif len(args.tasks) > 1:
        parser.error("--switch-at is required when multiple --tasks are provided")

    logger.info("-" * 40)
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg:<30}: {value}")
    logger.info("-" * 40)

    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset (e.g., 'username/dataset_name'):",
        )
        logger.info(f"Using repository ID: {args.repo_id}")

    if args.path is None:
        logger.info(" No path provided. Searching for models in 'outputs/train' directory.")

        # We check recursively in the 'outputs/train' directory for 'pretrained_model's recursively
        models_path = Path("outputs/train")
        if models_path.exists() and models_path.is_dir():
            models = [model for model in models_path.glob("**/pretrained_model") if model.is_dir()]
            models_names = sorted([str(model) for model in models], key=lambda x: x.lower())

            args.path = prompt.prompt(
                message="Please select a model to use for deployment:",
                options=models_names,
                default=models_names[0] if models else None,
            )
            logger.info(f"Using model path: {args.path}")
        else:
            logger.error("'outputs/models' directory does not exist.")
            logger.error(
                "Please provide a valid path to the model using --path or create a new one."
            )
            exit(1)

    if args.env_namespace is None:
        args.env_namespace = prompt.prompt(
            "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
            default="right",
        )
        logger.info(f"Using follower namespace: {args.env_namespace}")

    if args.env_config is None:
        follower_configs = list_env_configs()
        args.env_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info(f"Using follower configuration: {args.env_config}")

    if args.policy_config is None:
        policy_configs = list_policy_configs()
        args.policy_config = prompt.prompt(
            "Please select the policy configuration to use.",
            options=policy_configs,
            default=policy_configs[0],
        )

    if args.evaluate:
        logger.info("Evaluation mode enabled. Will evaluate the performance after each episode.")
        datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation_file = (
            prompt.prompt(
                "Please enter the output file for evaluation results",
                default=f"evaluation_results_{args.path.replace('/', '_')}_{datetime_now}",
            )
            + ".csv"
        )
    else:
        evaluation_file = "evaluation_results.csv"

    fiper_recorder_config = None
    if args.fiper_config is not None:
        fiper_recorder_config = load_fiper_recorder_config(config_path=Path(args.fiper_config))
        logger.info(f"Loaded FIPER config from {args.fiper_config}")

    fiper_output_dir = Path(args.fiper_output_dir) if args.fiper_output_dir is not None else None
    if fiper_recorder_config is not None and fiper_output_dir is None:
        parser.error("--fiper-output-dir is required when --fiper-config is set.")
    if fiper_output_dir is not None:
        logger.info(f"FIPER rollout files will be stored under {fiper_output_dir}")

    policy = None
    try:
        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(args.env_config, control_type=ctrl_type, namespace=args.env_namespace)

        # %% Prepare the dataset
        features = get_features(env)

        evaluator = Evaluator(output_file="eval/" + evaluation_file)

        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
            fiper_recording_enabled=fiper_recorder_config is not None,
        )
        recording_manager.wait_until_ready()

        # %% Set up multiprocessing for policy inference
        logger.info("Setting up the policy.")
        policy = make_policy(
            name_or_config_name=args.policy_config,
            pretrained_path=args.path,
            env=env,
            task=args.tasks[0],
            fiper_recorder_config=fiper_recorder_config,
            fiper_output_dir=fiper_output_dir,
        )
        if fiper_recorder_config is not None and not getattr(
            policy,
            "fiper_recording_enabled",
            False,
        ):
            raise RuntimeError(
                "FIPER rollout recording is currently supported only by lerobot_policy."
            )

        logger.info("Homing robot before starting with recording.")

        env.wait_until_ready()
        env.home()
        env.reset()

        task_switch_timers = []

        def schedule_task_switches():
            """Schedule timed task switches for the current episode."""
            for timer in task_switch_timers:
                timer.cancel()
            task_switch_timers.clear()

            policy.set_task(args.tasks[0])

            if args.switch_at is not None:
                for delay, task in zip(args.switch_at, args.tasks[1:]):
                    timer = threading.Timer(delay, policy.set_task, args=(task,))
                    timer.daemon = True
                    timer.start()
                    task_switch_timers.append(timer)

        def on_start():
            """Hook function to be called when starting a new episode."""
            env.reset()
            policy.reset()
            schedule_task_switches()
            evaluator.start_timer()

        def on_end():
            """Hook function to be called when stopping the recording."""
            for timer in task_switch_timers:
                timer.cancel()
            task_switch_timers.clear()
            env.robot.reset_targets()
            env.robot.home(blocking=False)
            env.gripper.open()

            logger.info("Waiting for user to decide on success/failure if evaluating...")
            if recording_manager.state != "exit":
                evaluator.evaluate(episode=recording_manager.episode_count)

        def on_save_fiper() -> bool:
            """Collect FIPER metadata and ask the policy worker to save rollout data."""
            metadata = collect_fiper_metadata()
            if metadata is None:
                return False
            policy.save_fiper_rollout(metadata)
            return True

        def on_delete_fiper() -> None:
            """Ask the policy worker to discard buffered FIPER rollout data."""
            policy.delete_fiper_rollout()

        fiper_on_save = on_save_fiper if fiper_recorder_config is not None else None
        fiper_on_delete = on_delete_fiper if fiper_recorder_config is not None else None

        with evaluator.start_eval(overwrite=True, activate=args.evaluate):
            with recording_manager:
                while not recording_manager.done():
                    logger.info(
                        f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
                    )

                    recording_manager.record_episode(
                        data_fn=policy.make_data_fn(),
                        task=args.tasks[0],
                        on_start=on_start,
                        on_end=on_end,
                        on_save=fiper_on_save,
                        on_delete=fiper_on_delete,
                        episode_len=args.episode_length,
                    )

                    logger.info("Episode finished.")

        # Shutdown inference process
        logger.info("Shutting down inference process.")
        policy.shutdown()

        logger.info("Homing robot.")
        env.home()

        logger.info("Closing the environment.")
        env.close()

        logger.info("Finished recording.")
    except Exception as e:
        logger.exception(e)
        if policy is not None:
            policy.shutdown()


if __name__ == "__main__":
    main()
