"""Backend interface plus fake and live implementations."""

from __future__ import annotations

import gzip
import json
import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from unet_inference_client import history, slicer_view
from unet_inference_client.config import Settings

GZIP_MAGIC = b"\x1f\x8b"
NRRD_MAGIC = b"NRRD"


@dataclass(frozen=True)
class Case:
    case_id: str
    has_ground_truth: bool


@dataclass(frozen=True)
class InferenceResult:
    case_id: str
    output_dir: Path
    prediction_path: Path
    elapsed_seconds: float
    epoch: str
    dice: str


class Backend:
    def server_status(self) -> tuple[bool, str]:
        raise NotImplementedError

    def list_cases(self) -> list[Case]:
        raise NotImplementedError

    def run_inference(self, case_id: str) -> InferenceResult:
        raise NotImplementedError

    def has_result(self, case_id: str) -> bool:
        raise NotImplementedError

    def open_in_slicer(self, case_id: str) -> None:
        raise NotImplementedError


class FakeBackend(Backend):
    """Runs the GUI with no server, GPU, or data."""

    def __init__(self, server_up: bool = True, delay_seconds: float = 3.0):
        self.server_up = server_up
        self.delay_seconds = delay_seconds
        self._output_root = Path(tempfile.gettempdir()) / "nsc_fake_results"
        self._output_root.mkdir(exist_ok=True)

    def server_status(self) -> tuple[bool, str]:
        if self.server_up:
            return True, "running (fake backend)"
        return False, "not running"

    def list_cases(self) -> list[Case]:
        return [Case(f"FAKE_CASE_{i:03d}", i % 3 != 0) for i in range(1, 9)]

    def run_inference(self, case_id: str) -> InferenceResult:
        if not self.server_up:
            raise RuntimeError("Server is not running.")
        started = time.monotonic()
        time.sleep(self.delay_seconds)
        output_dir = self._output_root / case_id
        output_dir.mkdir(exist_ok=True)
        prediction = output_dir / "pred.nrrd"
        prediction.write_text("fake prediction\n")
        return InferenceResult(
            case_id, output_dir, prediction, time.monotonic() - started, "fake", "fake"
        )

    def has_result(self, case_id: str) -> bool:
        return (self._output_root / case_id / "pred.nrrd").exists()

    def open_in_slicer(self, case_id: str) -> None:
        raise RuntimeError("Slicer is not available with the fake backend.")


class MonaiLabelBackend(Backend):
    """Talks to a live MONAI Label server."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def server_url(self) -> str:
        return self.settings.server_url.rstrip("/")

    def server_status(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.server_url}/info/", timeout=5)
            response.raise_for_status()
            info = response.json()
        except requests.RequestException as exc:
            return False, f"unreachable ({type(exc).__name__})"
        models = info.get("models", {})
        if self.settings.model not in models:
            available = ", ".join(models) or "none"
            return False, f"running, but '{self.settings.model}' not loaded (available: {available})"
        device = models[self.settings.model].get("config", {}).get("device", ["?"])[0]
        return True, f"running on {device}"

    def available_models(self) -> list[str]:
        response = requests.get(f"{self.server_url}/info/", timeout=10)
        response.raise_for_status()
        return sorted(response.json().get("models", {}))

    def list_cases(self) -> list[Case]:
        response = requests.get(
            f"{self.server_url}/datastore/", params={"output": "all"}, timeout=30
        )
        response.raise_for_status()
        objects = response.json().get("objects", {})
        cases = [
            Case(case_id, bool(entry.get("labels", {}).get("final")))
            for case_id, entry in objects.items()
        ]
        return sorted(cases, key=lambda c: c.case_id)

    def output_dir_for(self, case_id: str) -> Path:
        epoch, dice = self._checkpoint_metadata()
        name = f"{case_id}_{self.settings.resolution}_ep{epoch}_dice{dice}"
        return self.settings.results_path() / name

    def has_result(self, case_id: str) -> bool:
        return any(self.output_dir_for(case_id).glob("pred.*"))

    def run_inference(self, case_id: str) -> InferenceResult:
        is_up, message = self.server_status()
        if not is_up:
            raise RuntimeError(f"Server check failed: {message}")

        started = time.monotonic()
        try:
            response = requests.post(
                f"{self.server_url}/infer/{self.settings.model}",
                params={"image": case_id, "output": "image"},
                timeout=3600,
            )
            response.raise_for_status()
            payload = response.content
            if not payload:
                raise RuntimeError("Server returned an empty response.")
            if payload[:2] == GZIP_MAGIC:
                payload = gzip.decompress(payload)
            if payload[:4] != NRRD_MAGIC and payload[:2] not in (b"\x5c\x01", b"\x01\x5c"):
                preview = payload[:200].decode("utf-8", errors="replace")
                raise RuntimeError(f"Server returned non-image data: {preview}")
        except Exception as exc:
            history.record(
                self.settings.results_path(), case_id, "failed",
                time.monotonic() - started, self.settings.model, "", "", str(exc),
            )
            raise

        epoch, dice = self._checkpoint_metadata()
        output_dir = self.output_dir_for(case_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = "nrrd" if payload[:4] == NRRD_MAGIC else "nii"
        prediction = output_dir / f"pred.{extension}"
        prediction.write_bytes(payload)

        elapsed = time.monotonic() - started
        self._write_manifest(output_dir, case_id, extension, epoch, dice, elapsed)
        self._update_latest_symlink(output_dir)
        history.record(
            self.settings.results_path(), case_id, "completed", elapsed,
            self.settings.model, epoch, dice, str(output_dir),
        )

        return InferenceResult(case_id, output_dir, prediction, elapsed, epoch, dice)

    def open_in_slicer(self, case_id: str) -> None:
        predictions = sorted(self.output_dir_for(case_id).glob("pred.*"))
        if not predictions:
            raise RuntimeError(f"No prediction found for {case_id}")
        self.open_prediction(case_id, predictions[0])

    def open_prediction(self, case_id: str, prediction_path: Path) -> None:
        slicer_binary = self.settings.slicer_path()
        if slicer_binary is None:
            raise RuntimeError("Slicer path is not set. Configure it in Settings.")
        studies_dir = self.settings.studies_path()
        if studies_dir is None:
            raise RuntimeError("Studies folder is not set. Configure it in Settings.")
        image_path = slicer_view.find_source_image(studies_dir, case_id)
        if image_path is None:
            raise RuntimeError(f"No source image for {case_id} in {studies_dir}")
        slicer_view.launch(slicer_binary, image_path, prediction_path, case_id)

    def _checkpoint_metadata(self) -> tuple[str, str]:
        stats_file = self.settings.train_stats_path()
        if stats_file is None:
            return "unknown", "unknown"
        try:
            stats = json.loads(stats_file.read_text())
            return str(stats["eval"]["best_metric_epoch"]), f"{stats['eval']['best_metric']:.4f}"
        except (OSError, KeyError, ValueError, TypeError):
            return "unknown", "unknown"

    def _write_manifest(self, output_dir, case_id, extension, epoch, dice, elapsed) -> None:
        manifest = {
            "case_id": case_id,
            "model": self.settings.model,
            "resolution": self.settings.resolution,
            "checkpoint_epoch": epoch,
            "checkpoint_dice": dice,
            "prediction_format": extension,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server": self.server_url,
        }
        weights = self.settings.model_weights_path()
        if weights is not None and weights.exists():
            manifest["model_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _update_latest_symlink(self, output_dir: Path) -> None:
        link = self.settings.results_path() / "latest"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(output_dir)
