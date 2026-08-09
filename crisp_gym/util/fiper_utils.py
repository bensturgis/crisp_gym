import logging
import re  # noqa: D100
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from lerobot.fiper.data_generation.configuration_fiper_rollout_recorder import (
    FiperRolloutRecorderConfig,
)

from crisp_gym.util import prompt

logger = logging.getLogger(__name__)


def load_fiper_recorder_config(config_path: Path) -> FiperRolloutRecorderConfig:  # noqa: D103
    if not config_path.exists():
        logging.error(f"FIPER config file not found: {config_path}")
        raise FileNotFoundError(config_path)

    data = yaml.safe_load(config_path.read_text()) or {}
    allowed_fields = {field.name for field in fields(FiperRolloutRecorderConfig)}
    ignored_fields = sorted(set(data) - allowed_fields)
    if ignored_fields:
        logger.warning(
            "Ignoring obsolete FIPER rollout-recorder config fields: %s",
            ", ".join(ignored_fields),
        )
    fiper_recorder_config = FiperRolloutRecorderConfig(
        **{key: value for key, value in data.items() if key in allowed_fields}
    )

    return fiper_recorder_config


def next_fiper_episode_index(output_dir: Path) -> int:  # noqa: D103
    if not output_dir.exists():
        return 0

    pattern = re.compile(r"episode_[sf]_(\d{4})")

    max_idx = -1
    for file in output_dir.iterdir():
        if not file.is_file():
            continue

        match = pattern.search(file.stem)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1

def collect_fiper_metadata() -> dict[str, Any] | None:
    outcome = prompt.prompt(
        "Was this episode a success or failure?",
        options=["success", "failure"],
    ).lower()
    rollout_type = prompt.prompt(
        "Was this a calibration or test episode?",
        options=["calibration", "test"],
    ).lower()
    rollout_subtype = prompt.prompt(
        "Was this episode in-distribution or out-of-distribution?",
        options=["id", "ood"],
    ).lower()
    if rollout_type == "calibration" and not (rollout_subtype == "id" and outcome == "success"):
        logger.warning(
            "Not saving: Calibration episodes should be in-distribution and successful."
        )
        return None
    
    if rollout_type == "calibration":
        rollout_subtype = "ca"

    return {
        "metadata": True,
        "task": "lego_stacking",
        "successful": (outcome == "success"),
        "task_id": 0,
        "rollout_type": rollout_type,
        "rollout_subtype": rollout_subtype,
    }