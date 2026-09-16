"""User-configurable settings, persisted to disk."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "neural-segmentation-client"
CONFIG_FILE = CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    server_url: str = "http://127.0.0.1:8000"
    model: str = "deepedit_seg"
    results_dir: str = str(Path.home() / "segmentation-results")
    studies_dir: str = ""
    slicer_binary: str = ""
    model_dir: str = ""
    resolution: str = "192x192x224"
    skip_existing: bool = True

    @classmethod
    def load(cls) -> "Settings":
        if not CONFIG_FILE.exists():
            return cls()
        try:
            stored = json.loads(CONFIG_FILE.read_text())
        except (OSError, ValueError):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in stored.items() if k in known})

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    def results_path(self) -> Path:
        return Path(self.results_dir).expanduser()

    def studies_path(self) -> Path | None:
        return Path(self.studies_dir).expanduser() if self.studies_dir else None

    def slicer_path(self) -> Path | None:
        return Path(self.slicer_binary).expanduser() if self.slicer_binary else None

    def model_path(self) -> Path | None:
        return Path(self.model_dir).expanduser() if self.model_dir else None

    def train_stats_path(self) -> Path | None:
        model_dir = self.model_path()
        if model_dir is None:
            return None
        matches = sorted(model_dir.glob("*/train_stats.json"))
        return matches[0] if matches else None

    def model_weights_path(self) -> Path | None:
        model_dir = self.model_path()
        if model_dir is None:
            return None
        weights = sorted(model_dir.glob("*.pt"))
        return weights[0] if weights else None
