"""Properties Dice must satisfy on any pair of small random label maps."""

import numpy as np
import pytest
import SimpleITK as sitk
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from unet_inference_client import metrics

SHAPE = (4, 5, 6)
label_maps = npst.arrays(np.uint8, SHAPE, elements=st.integers(0, 2))


def write(tmp_path, name, array):
    path = tmp_path / f"{name}.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    return path


def dice(tmp_path, pred, gt):
    return metrics.compare(write(tmp_path, "pred", pred), write(tmp_path, "gt", gt), include_hd95=False).dice


# Module scope: every example overwrites the same two files, and Hypothesis rejects function-scoped fixtures.
@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("masks")


@settings(max_examples=40, deadline=None)
@given(pred=label_maps, gt=label_maps)
def test_dice_is_between_zero_and_one_and_symmetric(workdir, pred, gt):
    forward = dice(workdir, pred, gt)
    assert 0.0 <= forward <= 1.0
    assert forward == pytest.approx(dice(workdir, gt, pred))


@settings(max_examples=25, deadline=None)
@given(mask=label_maps)
def test_identical_masks_score_one(workdir, mask):
    assert dice(workdir, mask, mask) == 1.0


@settings(max_examples=25, deadline=None)
@given(mask=label_maps.filter(lambda a: a.any()))
def test_disjoint_masks_score_zero(workdir, mask):
    inverted = np.where(mask > 0, 0, 1).astype(np.uint8)
    assert dice(workdir, mask, inverted) == 0.0


@settings(max_examples=25, deadline=None)
@given(pred=label_maps, gt=label_maps)
def test_per_class_dice_matches_binary_dice_of_each_label(workdir, pred, gt):
    rows = metrics.compare_classes(write(workdir, "pred", pred), write(workdir, "gt", gt))
    expected = sorted(int(label) for label in set(np.unique(pred)) | set(np.unique(gt)) if label)
    assert [row.label for row in rows] == expected
    for row in rows:
        assert 0.0 <= row.dice <= 1.0
        single = metrics.compare(
            write(workdir, "pred", pred), write(workdir, "gt", gt), label=row.label, include_hd95=False
        )
        assert row.dice == pytest.approx(single.dice)
