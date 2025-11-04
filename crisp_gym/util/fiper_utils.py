import logging
import re  # noqa: D100
from pathlib import Path

import yaml
from lerobot.fiper_data_recorder.configuration_fiper_data_recorder import (
    FiperDataRecorderConfig,
    LaplaceConfig,
    LikelihoodODESolverConfig,
)


def load_fiper_recorder_config(config_path: Path) -> FiperDataRecorderConfig:  # noqa: D103
    if not config_path.exists():
        logging.error(f"FIPER config file not found: {config_path}")
        raise FileNotFoundError(config_path)

    data = yaml.safe_load(config_path.read_text())

    # Coerce fields to expected dataclass types
    if "scoring_metrics" in data and isinstance(data["scoring_metrics"], list):
        data["scoring_metrics"] = tuple(data["scoring_metrics"])

    if "laplace_config" in data and isinstance(data["laplace_config"], dict):
        data["laplace_config"] = LaplaceConfig(**data["laplace_config"])

    if "likelihood_ode_solver_cfg" in data and isinstance(data["likelihood_ode_solver_cfg"], dict):
        data["likelihood_ode_solver_cfg"] = LikelihoodODESolverConfig(**data["likelihood_ode_solver_cfg"])

    fiper_recorder_config = FiperDataRecorderConfig(**data)

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