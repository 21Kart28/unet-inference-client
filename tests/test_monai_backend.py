import csv
import json

import pytest
import SimpleITK as sitk

from unet_inference_client import history, server


def test_server_status_reports_device(backend):
    assert backend.server_status() == (True, "running on cuda:0")


def test_server_status_when_model_not_loaded(backend, mock_server):
    mock_server.models = ["segmentation_spleen"]
    is_up, message = backend.server_status()
    assert not is_up
    assert "'deepedit_seg' not loaded" in message
    assert "segmentation_spleen" in message


def test_server_status_when_unreachable(backend, mock_server):
    mock_server.stop()
    assert backend.server_status() == (False, "unreachable (ConnectionError)")


def test_is_responding(mock_server):
    assert server.is_responding(mock_server.url)
    mock_server.stop()
    assert not server.is_responding(mock_server.url)


def test_list_cases_sorted_with_ground_truth_flags(backend):
    cases = backend.list_cases()
    assert [case.case_id for case in cases] == ["CASE_001", "CASE_002", "CASE_003"]
    assert [case.has_ground_truth for case in cases] == [True, False, True]


def test_run_inference_saves_prediction_manifest_and_history(backend, settings, mock_server):
    result = backend.run_inference("CASE_001")

    assert mock_server.infer_calls == [("deepedit_seg", "CASE_001")]
    assert result.prediction_path.name == "pred.nrrd"
    assert result.output_dir.name == "CASE_001_192x192x224_epunknown_diceunknown"
    assert sitk.ReadImage(str(result.prediction_path)).GetSize() == (4, 4, 4)
    assert backend.has_result("CASE_001")

    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["case_id"] == "CASE_001"
    assert manifest["model"] == "deepedit_seg"
    assert manifest["server"] == mock_server.url

    with history.log_path(settings.results_path()).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["case_id"], row["status"]) for row in rows] == [("CASE_001", "completed")]


def test_run_inference_unpacks_gzipped_prediction(backend, mock_server):
    mock_server.gzip_output = True
    result = backend.run_inference("CASE_001")
    assert result.prediction_path.read_bytes()[:4] == b"NRRD"


def test_run_inference_rejects_non_image_response(backend, settings, mock_server):
    mock_server.infer_response = b'{"detail": "model crashed"}'
    with pytest.raises(RuntimeError, match="non-image data"):
        backend.run_inference("CASE_001")

    with history.log_path(settings.results_path()).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "failed"
    assert not backend.has_result("CASE_001")


def test_run_inference_fails_for_unknown_case(backend):
    with pytest.raises(Exception, match="404"):
        backend.run_inference("NOPE")


def test_run_inference_fails_when_model_missing(backend, mock_server):
    mock_server.models = []
    with pytest.raises(RuntimeError, match="Server check failed"):
        backend.run_inference("CASE_001")
