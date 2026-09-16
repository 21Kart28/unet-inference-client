"""Generates and launches a Slicer script to view a result."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

SCRIPT_TEMPLATE = '''import slicer

slicer.mrmlScene.Clear(0)
volume = slicer.util.loadVolume(r"{image_path}")
segmentation = slicer.util.loadSegmentation(r"{prediction_path}")

segmentation.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
segmentation.CreateClosedSurfaceRepresentation()
segmentation.GetDisplayNode().SetVisibility3D(True)
slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
print("[nsc] loaded {case_id} (surface smoothing disabled)")
'''

IMAGE_EXTENSIONS = (".nrrd", ".nii.gz", ".nii")


def find_source_image(studies_dir: Path, case_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = studies_dir / f"{case_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def launch(slicer_binary: Path, image_path: Path, prediction_path: Path, case_id: str) -> None:
    if not slicer_binary.exists():
        raise RuntimeError(f"Slicer not found at {slicer_binary}")
    if not prediction_path.exists():
        raise RuntimeError(f"Prediction not found at {prediction_path}")
    if not image_path.exists():
        raise RuntimeError(f"Source image not found at {image_path}")

    script = SCRIPT_TEMPLATE.format(
        image_path=image_path,
        prediction_path=prediction_path,
        case_id=case_id,
    )
    handle = tempfile.NamedTemporaryFile("w", suffix="_nsc_view.py", delete=False)
    handle.write(script)
    handle.close()

    subprocess.Popen([str(slicer_binary), "--python-script", handle.name])
