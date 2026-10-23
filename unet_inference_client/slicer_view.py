"""Generates and launches a Slicer script to view a result."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePath

SCRIPT_TEMPLATE = '''import slicer

slicer.mrmlScene.Clear(0)
volume = slicer.util.loadVolume({image_path!r})
segmentation = slicer.util.loadSegmentation({prediction_path!r})

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


def find_slicer() -> Path | None:
    found = [path for root, pattern in _install_globs() if root.is_dir() for path in root.glob(pattern)]
    on_path = shutil.which("Slicer")
    if on_path:
        found.append(Path(on_path))
    return max(found, key=_version, default=None)


def _install_globs() -> list[tuple[Path, str]]:
    if sys.platform == "win32":
        roots = [Path.home()]
        if os.environ.get("LOCALAPPDATA"):
            roots.append(Path(os.environ["LOCALAPPDATA"]) / "slicer.org")
        if os.environ.get("ProgramFiles"):
            roots.append(Path(os.environ["ProgramFiles"]))
        return [(root, "*Slicer*/Slicer.exe") for root in roots]
    if sys.platform == "darwin":
        return [(Path("/Applications"), "Slicer*.app/Contents/MacOS/Slicer")]
    return [(Path.home(), "Slicer*/Slicer"), (Path("/opt"), "Slicer*/Slicer")]


def _version(path: Path) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", path.parent.name)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def build_script(image_path: PurePath, prediction_path: PurePath, case_id: str) -> str:
    # Slicer accepts forward slashes on every OS.
    return SCRIPT_TEMPLATE.format(
        image_path=image_path.as_posix(),
        prediction_path=prediction_path.as_posix(),
        case_id=case_id,
    )


def launch(slicer_binary: Path, image_path: Path, prediction_path: Path, case_id: str) -> None:
    if not slicer_binary.exists():
        raise RuntimeError(
            f"Slicer not found at {slicer_binary}. Fix the Slicer executable in Settings,"
            " or clear it to detect Slicer automatically."
        )
    if not prediction_path.exists():
        raise RuntimeError(f"Prediction not found at {prediction_path}")
    if not image_path.exists():
        raise RuntimeError(f"Source image not found at {image_path}")

    handle = tempfile.NamedTemporaryFile("w", suffix="_nsc_view.py", delete=False, encoding="utf-8")
    handle.write(build_script(image_path, prediction_path, case_id))
    handle.close()

    subprocess.Popen([str(slicer_binary), "--python-script", handle.name])
