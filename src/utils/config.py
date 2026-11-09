from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("defaults", None)
    if parent:
        base_path = Path(parent)
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        base = load_config(str(base_path))
        base.update(config)
        return base
    return config
