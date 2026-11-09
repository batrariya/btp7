from __future__ import annotations
import logging
from pathlib import Path


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("fairskew")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
