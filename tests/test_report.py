from unet_inference_client import metrics, model_compare, report
from unet_inference_client.history import StoredResult


def stored(tmp_path, case_id, model, dice, classes=None):
    folder = tmp_path / f"{case_id}_{model}"
    folder.mkdir()
    metrics.save_metrics(
        folder, metrics.MaskMetrics(dice, 10.0, 12.0, -16.7, 3.2), "gt.nrrd", classes=classes
    )
    return StoredResult(case_id, folder, folder / "pred.nrrd", model, "40", "0.91", "2026-01-01")


def test_summary_stats():
    stats = dict(report.summary_stats([0.9, 0.7, 0.8]))
    assert stats["Cases"] == "3"
    assert stats["Mean Dice"] == "0.8000"
    assert stats["Median Dice"] == "0.8000"
    assert stats["Min Dice"] == "0.7000"
    assert report.summary_stats([]) == []


def test_batch_report_lists_cases_and_classes(tmp_path):
    classes = [metrics.ClassMetrics(1, 0.95, 5.0, 5.0), metrics.ClassMetrics(2, 0.85, 5.0, 7.0)]
    results = [
        stored(tmp_path, "CASE_001", "unet", 0.9, classes),
        stored(tmp_path, "CASE_002", "unet", 0.7),
    ]
    (tmp_path / "no_metrics").mkdir()
    results.append(StoredResult("CASE_003", tmp_path / "no_metrics", tmp_path / "pred.nrrd", "unet", "?", "?", ""))

    document = report.from_results(results, "http://lab:8000", "unet")
    assert ("Server", "http://lab:8000") in document.details
    assert ("Mean Dice", "0.8000") in document.details
    cases, classes_table = document.tables
    assert [row[0] for row in cases.rows] == ["CASE_001", "CASE_002"]
    assert cases.rows[0][3] == "0.9000"
    assert cases.rows[0][7] == "3.20"
    assert classes_table.rows == [
        ["CASE_001", "1", "0.9500", "5.0", "5.0"],
        ["CASE_001", "2", "0.8500", "5.0", "7.0"],
    ]
    assert document.notes == []


def test_batch_report_without_metrics_says_what_to_do(tmp_path):
    document = report.from_results([], "http://lab:8000", "unet")
    assert len(document.tables) == 1
    assert document.tables[0].rows == []
    assert document.notes[0].startswith("No result has saved metrics.")


def test_markdown_has_table_and_details():
    document = report.Report(
        "Batch metrics report", [("Model", "unet")],
        [report.Table("Per-case metrics", ["Case", "Dice"], [["CASE_001", "0.9000"], ["CASE_002", ""]])],
    )
    text = report.to_markdown(document)
    assert text.startswith("# Batch metrics report\n\n- **Model:** unet\n")
    assert "| Case | Dice |\n| --- | --- |\n| CASE_001 | 0.9000 |\n| CASE_002 | — |\n" in text


def test_html_escapes_and_renders_rows():
    document = report.Report(
        "Report <1>", [("Server", "http://a?b=1&c=2")],
        [report.Table("Cases", ["Case"], [["<CASE>"]])], notes=["All good & fine."],
    )
    text = report.to_html(document)
    assert "<title>Report &lt;1&gt;</title>" in text
    assert "<dd>http://a?b=1&amp;c=2</dd>" in text
    assert "<td>&lt;CASE&gt;</td>" in text
    assert "<li>All good &amp; fine.</li>" in text


def test_comparison_report_uses_signed_deltas():
    rows = [model_compare.DeltaRow("CASE_001", 0.8, 0.9), model_compare.DeltaRow("CASE_002", 0.7, None)]
    document = report.from_deltas(rows, model_compare.summarize(rows), "unet", "unet_v2", "http://lab")
    assert document.tables[0].headers == ["Case", "unet Dice", "unet_v2 Dice", "Delta"]
    assert document.tables[0].rows[0] == ["CASE_001", "0.8000", "0.9000", "+0.1000"]
    assert document.tables[0].rows[1][3] == ""
    assert ("Mean delta", "+0.1000") in document.details
    assert document.notes[0].startswith("1 paired case(s): unet_v2 is better")


def test_write_picks_format_from_suffix(tmp_path):
    document = report.Report("T", [], [report.Table("X", ["A"], [["1"]])])
    report.write(tmp_path / "r.md", document)
    report.write(tmp_path / "r.HTML", document)
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# T")
    assert (tmp_path / "r.HTML").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
