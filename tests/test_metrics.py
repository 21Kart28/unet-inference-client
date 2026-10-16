import numpy as np
import pytest
import SimpleITK as sitk

from unet_inference_client import metrics
from unet_inference_client.metrics import GeometryMismatch


def make_mask(size=(30, 30, 30), boxes=(), spacing=(1.0, 1.0, 1.0)):
    """Each box is ((x, y, z) start, (x, y, z) stop, value)."""
    array = np.zeros(size[::-1], dtype=np.uint8)
    for start, stop, value in boxes:
        array[start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return image


def cube(start, edge=10, value=1):
    stop = tuple(s + edge for s in start)
    return (start, stop, value)


@pytest.fixture
def write(tmp_path):
    def _write(name, image):
        path = tmp_path / f"{name}.nrrd"
        sitk.WriteImage(image, str(path))
        return path
    return _write


def test_identical_masks(write):
    mask = make_mask(boxes=[cube((5, 5, 5))])
    result = metrics.compare(write("pred", mask), write("gt", mask))
    assert result.dice == 1.0
    assert result.hd95_mm == 0.0
    assert result.volume_diff_pct == 0.0
    assert result.note == ""


def test_cube_volume_uses_spacing(write):
    mask = make_mask(boxes=[cube((5, 5, 5))], spacing=(0.5, 0.5, 0.5))
    assert metrics.volume_mm3(write("cube", mask)) == 125.0


def test_overlapping_cubes_dice(write):
    pred = make_mask(boxes=[cube((5, 5, 5))])
    gt = make_mask(boxes=[cube((8, 7, 5))])
    result = metrics.compare(write("pred", pred), write("gt", gt))
    # Overlap is 7 x 8 x 10 voxels; each cube has 1000.
    assert result.dice == pytest.approx(2 * 560 / 2000)


def test_one_voxel_shift_hd95(write):
    spacing = (0.5, 0.5, 0.5)
    pred = make_mask(boxes=[cube((5, 5, 5))], spacing=spacing)
    gt = make_mask(boxes=[cube((6, 5, 5))], spacing=spacing)
    result = metrics.compare(write("pred", pred), write("gt", gt))
    assert result.hd95_mm == pytest.approx(0.5)


def test_hd95_unchanged_by_crop(write):
    small_pred = make_mask(boxes=[cube((5, 5, 5))])
    small_gt = make_mask(boxes=[cube((7, 6, 5), edge=12)])
    large_pred = make_mask(size=(200, 200, 200), boxes=[cube((150, 40, 90))])
    large_gt = make_mask(size=(200, 200, 200), boxes=[cube((152, 41, 90), edge=12)])
    small = metrics.compare(write("small_pred", small_pred), write("small_gt", small_gt))
    large = metrics.compare(write("large_pred", large_pred), write("large_gt", large_gt))
    assert large.hd95_mm == pytest.approx(small.hd95_mm)
    assert large.dice == pytest.approx(small.dice)


def test_hd95_can_be_skipped(write):
    mask = make_mask(boxes=[cube((5, 5, 5))])
    result = metrics.compare(write("pred", mask), write("gt", mask), include_hd95=False)
    assert result.hd95_mm is None
    assert result.dice == 1.0


def test_mismatched_spacing_raises(write):
    pred = make_mask(boxes=[cube((5, 5, 5))], spacing=(0.5, 0.5, 0.5))
    gt = make_mask(boxes=[cube((5, 5, 5))], spacing=(1.0, 1.0, 1.0))
    with pytest.raises(GeometryMismatch, match="Spacing differs.*0.5.*ground truth \\(1, 1, 1\\)"):
        metrics.compare(write("pred", pred), write("gt", gt))


def test_mismatched_size_raises(write):
    pred = make_mask(size=(30, 30, 30), boxes=[cube((5, 5, 5))])
    gt = make_mask(size=(30, 30, 31), boxes=[cube((5, 5, 5))])
    with pytest.raises(GeometryMismatch, match="Size differs"):
        metrics.compare(write("pred", pred), write("gt", gt))


def test_spacing_within_tolerance_is_accepted():
    pred = make_mask(spacing=(0.5, 0.5, 0.5))
    gt = make_mask(spacing=(0.50000005, 0.5, 0.5))
    metrics.check_same_geometry(pred, gt)


def test_mismatched_origin_raises():
    pred = make_mask()
    gt = make_mask()
    gt.SetOrigin((0.0, 0.0, 2.5))
    with pytest.raises(GeometryMismatch, match="Origin differs"):
        metrics.check_same_geometry(pred, gt)


def test_label_selects_one_value(write):
    image = make_mask(boxes=[cube((2, 2, 2), edge=5, value=1), cube((15, 15, 15), edge=4, value=2)])
    path = write("labels", image)
    assert metrics.volume_mm3(path, label=2) == 64.0
    assert metrics.volume_mm3(path) == 125.0 + 64.0


def test_label_applies_to_compare(write):
    pred = make_mask(boxes=[cube((2, 2, 2), edge=5, value=1), cube((15, 15, 15), edge=4, value=2)])
    gt = make_mask(boxes=[cube((15, 15, 15), edge=4, value=2)])
    result = metrics.compare(write("pred", pred), write("gt", gt), label=2)
    assert result.dice == 1.0


def test_both_empty(write):
    empty = make_mask()
    result = metrics.compare(write("pred", empty), write("gt", empty))
    assert result.dice == 1.0
    assert result.hd95_mm is None
    assert result.volume_diff_pct is None
    assert result.note == "Both masks are empty."


def test_prediction_empty(write):
    result = metrics.compare(write("pred", make_mask()), write("gt", make_mask(boxes=[cube((5, 5, 5))])))
    assert result.dice == 0.0
    assert result.hd95_mm is None
    assert result.volume_diff_pct == -100.0
    assert "prediction mask is empty" in result.note


def test_ground_truth_empty(write):
    result = metrics.compare(write("pred", make_mask(boxes=[cube((5, 5, 5))])), write("gt", make_mask()))
    assert result.dice == 0.0
    assert result.hd95_mm is None
    assert result.volume_diff_pct is None
    assert "ground truth mask is empty" in result.note


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.nrrd"):
        metrics.load_mask(tmp_path / "missing.nrrd")


def test_find_ground_truth(tmp_path):
    final = tmp_path / "labels" / "final"
    final.mkdir(parents=True)
    (final / "CASE_1.nii.gz").write_bytes(b"")
    assert metrics.find_ground_truth(tmp_path, "CASE_1") == final / "CASE_1.nii.gz"
    assert metrics.find_ground_truth(tmp_path, "CASE_2") is None


def test_save_and_load_metrics(tmp_path):
    result = metrics.MaskMetrics(0.9, 100.0, 110.0, -9.09, 1.5)
    metrics.save_metrics(tmp_path, result, tmp_path / "gt.nrrd", label=2)
    saved = metrics.load_metrics(tmp_path)
    assert saved["dice"] == 0.9
    assert saved["hd95_mm"] == 1.5
    assert saved["label"] == 2
    assert saved["ground_truth_path"] == str(tmp_path / "gt.nrrd")
    assert metrics.load_metrics(tmp_path / "nowhere") is None
