"""Utilities for FIPER rollout recording."""

from __future__ import annotations

import logging
import re
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import yaml

from crisp_gym.util import prompt

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _get_fiper_rollout_recorder_config_cls() -> type:
    try:
        from lerobot.fiper.data_generation.configuration_fiper_rollout_recorder import (
            FiperRolloutRecorderConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "FIPER rollout recording requires a LeRobot install that includes "
            "`lerobot.fiper`. Install the relevant LeRobot extras before using "
            "`--fiper-config`."
        ) from exc

    return FiperRolloutRecorderConfig


def load_fiper_recorder_config(config_path: Path) -> Any:
    """Load a FIPER rollout recorder config from YAML."""
    if not config_path.exists():
        logger.error(f"FIPER config file not found: {config_path}")
        raise FileNotFoundError(config_path)

    config_cls = _get_fiper_rollout_recorder_config_cls()
    data = yaml.safe_load(config_path.read_text()) or {}
    allowed_fields = {field.name for field in fields(config_cls)}
    ignored_fields = sorted(set(data) - allowed_fields)
    if ignored_fields:
        logger.warning(
            "Ignoring obsolete FIPER rollout-recorder config fields: %s",
            ", ".join(ignored_fields),
        )

    return config_cls(**{key: value for key, value in data.items() if key in allowed_fields})


def next_fiper_episode_index(output_dir: Path) -> int:
    """Return the next episode index for a FIPER rollout output directory."""
    if not output_dir.exists():
        return 0

    pattern = re.compile(r"episode_[sf]_(\d{4})")
    max_idx = -1
    for file in output_dir.iterdir():
        if not file.is_file():
            continue

        match = pattern.search(file.stem)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    return max_idx + 1


def collect_fiper_metadata(
    task: str = "lego_stacking",
    task_id: int = 0,
) -> dict[str, Any] | None:
    """Prompt for FIPER episode metadata.

    Calibration episodes are only saved when they are successful and
    in-distribution, matching the FIPER rollout recorder's expected layout.
    """
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

    if rollout_type == "calibration" and not (
        rollout_subtype == "id" and outcome == "success"
    ):
        logger.warning("Not saving: Calibration episodes should be in-distribution and successful.")
        return None

    if rollout_type == "calibration":
        rollout_subtype = "ca"

    return {
        "metadata": True,
        "task": task,
        "successful": outcome == "success",
        "task_id": task_id,
        "rollout_type": rollout_type,
        "rollout_subtype": rollout_subtype,
    }
