"""One logger format for every script, written to console and optionally a file.

Using ``logging`` instead of ``print`` gives timestamps and levels, and lets a
training run save its console log next to its checkpoints.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "fracdet", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate lines via the root logger (Jupyter sets one)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    # Calling get_logger twice (common in notebooks) must not add a second handler,
    # otherwise every message prints twice.
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in logger.handlers
    ):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            for h in logger.handlers
        )
        if not already:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
