from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    paths = [
        Path(config["data"]["train_path"]).parent,
        Path(config["data"]["holdout_path"]).parent,
        Path(config["artifacts"]["model_dir"]),
        Path(config["artifacts"]["report_dir"]),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
