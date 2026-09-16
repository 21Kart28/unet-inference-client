"""Scans the results folder and records a CSV of every inference attempt."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

LOG_NAME = "inference_history.csv"
FIELDS = ["timestamp", "case_id", "status", "elapsed_seconds", "model", "epoch", "dice", "output_dir"]


@dataclass(frozen=True)
class StoredResult:
    case_id: str
    output_dir: Path
    prediction_path: Path
    model: str
    epoch: str
    dice: str
    timestamp: str


def scan(results_dir: Path) -> list[StoredResult]:
    if not results_dir.is_dir():
        return []
    found = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        predictions = sorted(entry.glob("pred.*"))
        if not predictions:
            continue
        found.append(_read_metadata(entry, predictions[0]))
    return found


def _read_metadata(output_dir: Path, prediction: Path) -> StoredResult:
    data = {}
    json_manifest = output_dir / "manifest.json"
    text_manifest = output_dir / "manifest.txt"
    if json_manifest.exists():
        try:
            data = json.loads(json_manifest.read_text())
        except ValueError:
            data = {}
    elif text_manifest.exists():
        for line in text_manifest.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

    modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(prediction.stat().st_mtime))
    return StoredResult(
        case_id=data.get("case_id", output_dir.name),
        output_dir=output_dir,
        prediction_path=prediction,
        model=data.get("model", "unknown"),
        epoch=str(data.get("checkpoint_epoch", "?")),
        dice=str(data.get("checkpoint_dice", "?")),
        timestamp=data.get("timestamp", modified),
    )


def record(results_dir: Path, case_id: str, status: str, elapsed: float,
           model: str, epoch: str, dice: str, output_dir: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / LOG_NAME
    is_new = not log_file.exists()
    with log_file.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "model": model,
            "epoch": epoch,
            "dice": dice,
            "output_dir": output_dir,
        })


def log_path(results_dir: Path) -> Path:
    return results_dir / LOG_NAME
