import pytest

from unet_inference_client import metrics
from unet_inference_client.history import StoredResult
from unet_inference_client.model_compare_tab import ModelCompareTab, format_delta


@pytest.fixture
def tab(qt_app, settings):
    return ModelCompareTab(settings, lambda text: None)


def stored(tmp_path, case_id, model, dice):
    folder = tmp_path / f"{case_id}_{model}"
    folder.mkdir()
    metrics.save_metrics(folder, metrics.MaskMetrics(dice, 1.0, 1.0, 0.0, None), "gt.nrrd")
    return StoredResult(case_id, folder, folder / "pred.nrrd", model, "1", "?", "2026-01-01")


def test_empty_results_explain_what_to_do(tab):
    tab.set_results([])
    assert tab.table.rowCount() == 0
    assert tab.summary_label.text().startswith("No saved Dice yet.")


def test_two_models_fill_the_table_and_summary(tab, tmp_path):
    tab.set_results([
        stored(tmp_path, "CASE_001", "unet", 0.8),
        stored(tmp_path, "CASE_001", "unet_v2", 0.9),
        stored(tmp_path, "CASE_002", "unet", 0.7),
    ])
    assert tab.model_a_box.currentText() == "unet"
    assert tab.model_b_box.currentText() == "unet_v2"
    assert tab.table.rowCount() == 2
    assert tab.table.item(0, 3).text() == "+0.1000"
    assert tab.table.item(1, 3).text() == "\u2014"
    assert tab.summary_label.text().startswith("1 paired case(s): unet_v2 is better on average (+0.1000 Dice)")


def test_format_delta():
    assert format_delta(None) == ""
    assert format_delta(-0.05) == "-0.0500"
