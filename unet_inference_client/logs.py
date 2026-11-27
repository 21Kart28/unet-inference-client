"""Rotating log file under the config folder, so a failed lab session can be read back later."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from unet_inference_client.config import CONFIG_DIR

LOG_DIR = CONFIG_DIR / "logs"
LOG_NAME = "unet-client.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

logger = logging.getLogger("unet_inference_client")


def configure(log_dir: Path = LOG_DIR) -> Path | None:
    """Attach a rotating file handler once; returns the log path, or None if the folder is not writable."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_NAME
    if any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in logger.handlers):
        return path
    try:
        handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path
