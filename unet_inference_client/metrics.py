"""Overlap, volume and surface-distance metrics for segmentation masks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

GEOMETRY_TOLERANCE = 1e-4
CROP_MARGIN = 2
METRICS_NAME = "metrics.json"
LABEL_EXTENSIONS = (".nrrd", ".nii.gz", ".nii")


class GeometryMismatch(Exception):
    """The prediction and ground truth are not on the same voxel grid."""


@dataclass(frozen=True)
class MaskMetrics:
    dice: float
    pred_volume_mm3: float
    gt_volume_mm3: float
    volume_diff_pct: float | None
    hd95_mm: float | None
    note: str = ""


def load_mask(path: Path | str, label: int | None = None) -> sitk.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError:
        raise ValueError(f"Could not read {path.name} as an image.") from None
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{path.name} is not a single 3D volume.")
    if label is None:
        return sitk.NotEqual(image, 0)
    return sitk.Equal(image, label)


def volume_mm3(path: Path | str, label: int | None = None) -> float:
    mask = load_mask(path, label)
    return _count(mask) * _voxel_volume(mask)


def check_same_geometry(pred: sitk.Image, gt: sitk.Image) -> None:
    if pred.GetSize() != gt.GetSize():
        raise GeometryMismatch(_mismatch("Size", pred.GetSize(), gt.GetSize()))
    checks = (
        ("Spacing", pred.GetSpacing(), gt.GetSpacing()),
        ("Origin", pred.GetOrigin(), gt.GetOrigin()),
        ("Direction", pred.GetDirection(), gt.GetDirection()),
    )
    for name, pred_value, gt_value in checks:
        if not np.allclose(pred_value, gt_value, rtol=0, atol=GEOMETRY_TOLERANCE):
            raise GeometryMismatch(_mismatch(name, pred_value, gt_value))


def compare(pred_path: Path | str, gt_path: Path | str, label: int | None = None,
            include_hd95: bool = True) -> MaskMetrics:
    pred = load_mask(pred_path, label)
    gt = load_mask(gt_path, label)
    check_same_geometry(pred, gt)

    voxel = _voxel_volume(pred)
    pred_count, gt_count = _count(pred), _count(gt)
    pred_volume, gt_volume = pred_count * voxel, gt_count * voxel
    volume_diff = (pred_volume - gt_volume) / gt_volume * 100 if gt_count else None

    if not pred_count and not gt_count:
        return MaskMetrics(1.0, 0.0, 0.0, None, None, "Both masks are empty.")
    if not pred_count:
        note = "The prediction mask is empty, so there is no HD95."
        return MaskMetrics(0.0, pred_volume, gt_volume, volume_diff, None, note)
    if not gt_count:
        note = "The ground truth mask is empty, so there is no HD95 or volume difference."
        return MaskMetrics(0.0, pred_volume, gt_volume, volume_diff, None, note)

    # Full scans can be 704x704x992; everything below runs on the region around the masks.
    pred, gt = _crop_to_union(pred, gt)
    overlap = int(np.count_nonzero(np.logical_and(_array(pred), _array(gt))))
    dice = 2 * overlap / (pred_count + gt_count)
    hd95 = _hd95(pred, gt) if include_hd95 else None
    return MaskMetrics(dice, pred_volume, gt_volume, volume_diff, hd95)


def find_ground_truth(studies_dir: Path | str, case_id: str) -> Path | None:
    folder = Path(studies_dir) / "labels" / "final"
    for extension in LABEL_EXTENSIONS:
        candidate = folder / f"{case_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def save_metrics(result_dir: Path | str, metrics: MaskMetrics, gt_path: Path | str,
                 label: int | None = None) -> None:
    data = asdict(metrics) | {
        "ground_truth_path": str(gt_path),
        "label": label,
        "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (Path(result_dir) / METRICS_NAME).write_text(json.dumps(data, indent=2))


def load_metrics(result_dir: Path | str) -> dict | None:
    path = Path(result_dir) / METRICS_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayViewFromImage(image)


def _count(mask: sitk.Image) -> int:
    return int(np.count_nonzero(_array(mask)))


def _voxel_volume(image: sitk.Image) -> float:
    return float(np.prod(image.GetSpacing()))


def _mismatch(name: str, pred_value, gt_value) -> str:
    return f"{name} differs: prediction {_format(pred_value)}, ground truth {_format(gt_value)}."


def _format(values) -> str:
    return "(" + ", ".join(f"{value:.10g}" for value in values) + ")"


def _bounds(array: np.ndarray) -> tuple[list[int], list[int]]:
    lower, upper = [], []
    for axis in range(array.ndim):
        others = tuple(other for other in range(array.ndim) if other != axis)
        hits = np.flatnonzero(array.any(axis=others))
        lower.append(int(hits[0]))
        upper.append(int(hits[-1]))
    return lower, upper


def _crop_to_union(pred: sitk.Image, gt: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    pred_lower, pred_upper = _bounds(_array(pred))
    gt_lower, gt_upper = _bounds(_array(gt))
    # numpy axes run (z, y, x); SimpleITK indexes (x, y, z).
    lower = [min(a, b) for a, b in zip(pred_lower, gt_lower)][::-1]
    upper = [max(a, b) for a, b in zip(pred_upper, gt_upper)][::-1]
    region = tuple(
        slice(max(0, low - CROP_MARGIN), min(size, high + 1 + CROP_MARGIN))
        for low, high, size in zip(lower, upper, pred.GetSize())
    )
    return pred[region], gt[region]


def _surface_distances(source: sitk.Image, target: sitk.Image) -> np.ndarray:
    """Distance in mm from each surface voxel of `source` to the surface of `target`."""
    surface = sitk.GetArrayFromImage(sitk.LabelContour(source)).astype(bool)
    distance = sitk.SignedMaurerDistanceMap(target, squaredDistance=False, useImageSpacing=True)
    return np.abs(sitk.GetArrayFromImage(distance)[surface])


def _hd95(pred: sitk.Image, gt: sitk.Image) -> float:
    distances = np.concatenate([_surface_distances(pred, gt), _surface_distances(gt, pred)])
    return float(np.percentile(distances, 95))
