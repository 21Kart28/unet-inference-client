import csv

import pytest

from unet_inference_client import metrics, model_compare
from unet_inference_client.history import StoredResult


@pytest.fixture
def results(tmp_path):
    def stored(case_id, model, dice, timestamp="2026-01-01 10:00:00"):
        folder = tmp_path / f"{case_id}_{model}_{timestamp[:10]}"
        folder.mkdir()
        if dice is not None:
            metrics.save_metrics(folder, metrics.MaskMetrics(dice, 1.0, 1.0, 0.0, None), "gt.nrrd")
        return StoredResult(case_id, folder, folder / "pred.nrrd", model, "1", "?", timestamp)

    return [
        stored("CASE_001", "unet", 0.80),
        stored("CASE_001", "unet_v2", 0.85),
        stored("CASE_002", "unet", 0.90),
        stored("CASE_002", "unet_v2", 0.70),
        stored("CASE_003", "unet", 0.60),
        stored("CASE_003", "unet", 0.65, timestamp="2026-01-02 10:00:00"),
        stored("CASE_004", "unet_v2", None),
    ]


def test_dice_by_model_keeps_the_newest_result_per_case(results):
    table = model_compare.dice_by_model(results)
    assert table["unet"] == {"CASE_001": 0.80, "CASE_002": 0.90, "CASE_003": 0.65}
    assert table["unet_v2"] == {"CASE_001": 0.85, "CASE_002": 0.70}


def test_delta_rows_cover_the_union_of_cases(results):
    rows = model_compare.delta_rows(results, "unet", "unet_v2")
    assert [row.case_id for row in rows] == ["CASE_001", "CASE_002", "CASE_003"]
    assert rows[0].delta == pytest.approx(0.05)
    assert rows[1].delta == pytest.approx(-0.20)
    assert rows[2].dice_b is None and rows[2].delta is None


def test_summary_counts_paired_cases(results):
    summary = model_compare.summarize(model_compare.delta_rows(results, "unet", "unet_v2"))
    assert (summary.paired, summary.better, summary.worse) == (2, 1, 1)
    assert summary.mean_delta == pytest.approx(-0.075)
    text = model_compare.describe(summary, "unet", "unet_v2")
    assert text.startswith("2 paired case(s): unet_v2 is worse on average (-0.0750 Dice)")


def test_summary_with_nothing_paired():
    summary = model_compare.summarize([model_compare.DeltaRow("A", 0.5, None)])
    assert summary == model_compare.DeltaSummary(0, 0, 0, None)
    assert "No case has saved Dice" in model_compare.describe(summary, "a", "b")


def test_write_delta_csv(results, tmp_path):
    rows = model_compare.delta_rows(results, "unet", "unet_v2")
    target = tmp_path / "deltas.csv"
    model_compare.write_delta_csv(target, rows, "unet", "unet_v2")
    with target.open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["case_id"] == "CASE_001"
    assert float(written[0]["delta"]) == pytest.approx(0.05)
    assert written[2]["delta"] == ""
