import csv
import json
from pathlib import Path

import pytest
import SimpleITK as sitk

from unet_inference_client import history, server, slicer_view


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
    assert result.output_dir.name == "CASE_001_192x192x224_ep150_dice0.9123"
    assert sitk.ReadImage(str(result.prediction_path)).GetSize() == (4, 4, 4)
    assert backend.has_result("CASE_001")

    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert manifest["case_id"] == "CASE_001"
    assert manifest["model"] == "deepedit_seg"
    assert manifest["checkpoint_epoch"] == "150"
    assert manifest["checkpoint_dice"] == "0.9123"
    assert manifest["server"] == mock_server.url
    assert history.scan(settings.results_path())[0].server == mock_server.url

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


def test_snapshot_gathers_status_cases_and_stats(backend):
    snapshot = backend.snapshot()
    assert snapshot.is_up
    assert snapshot.message == "running on cuda:0"
    assert [case.case_id for case in snapshot.cases] == ["CASE_001", "CASE_002", "CASE_003"]
    assert (snapshot.total_images, snapshot.labeled_images) == (3, 2)
    assert (snapshot.epoch, snapshot.dice) == ("150", "0.9123")


def test_snapshot_when_unreachable(backend, mock_server):
    mock_server.stop()
    snapshot = backend.snapshot()
    assert not snapshot.is_up
    assert snapshot.message.startswith("unreachable")
    assert snapshot.cases == []


def test_checkpoint_unknown_without_train_stats(backend, mock_server):
    mock_server.train_stats = {}
    snapshot = backend.snapshot()
    assert (snapshot.epoch, snapshot.dice) == ("unknown", "unknown")
    assert backend.output_dir_for("CASE_001").name == "CASE_001_192x192x224_epunknown_diceunknown"


def test_checkpoint_uses_single_trainer_with_other_name(backend, mock_server):
    mock_server.train_stats = {"deepedit": {"eval": {"best_metric": 0.5, "best_metric_epoch": 7}}}
    snapshot = backend.snapshot()
    assert (snapshot.epoch, snapshot.dice) == ("7", "0.5000")


def test_checkpoint_falls_back_to_local_train_stats(backend, settings, mock_server, tmp_path):
    mock_server.train_stats = {}
    model_dir = tmp_path / "model"
    (model_dir / "run1").mkdir(parents=True)
    (model_dir / "run1" / "train_stats.json").write_text(
        json.dumps({"eval": {"best_metric": 0.8, "best_metric_epoch": 42}})
    )
    settings.model_dir = str(model_dir)
    assert backend.snapshot().epoch == "42"
    assert backend.output_dir_for("CASE_001").name == "CASE_001_192x192x224_ep42_dice0.8000"


def test_download_image_caches_under_results(backend, settings):
    path = backend.download_image("CASE_001")
    assert path == settings.results_path() / "_images" / "CASE_001.nrrd"
    assert path.read_bytes()[:4] == b"NRRD"


def test_download_image_fails_for_unknown_case(backend):
    with pytest.raises(Exception, match="404"):
        backend.download_image("NOPE")


def test_open_prediction_downloads_image_when_no_studies_folder(backend, settings, monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr(slicer_view, "launch", lambda *args: launched.append(args))
    settings.studies_dir = ""
    settings.slicer_binary = str(tmp_path / "Slicer.exe")
    prediction = tmp_path / "pred.nrrd"

    backend.open_prediction("CASE_001", prediction)
    backend.open_prediction("CASE_001", prediction)

    image = settings.results_path() / "_images" / "CASE_001.nrrd"
    assert launched == [(Path(settings.slicer_binary), image, prediction, "CASE_001")] * 2


def test_open_prediction_prefers_local_studies_image(backend, settings, monkeypatch, tmp_path):
    launched = []
    monkeypatch.setattr(slicer_view, "launch", lambda *args: launched.append(args))
    settings.slicer_binary = str(tmp_path / "Slicer.exe")
    studies = settings.studies_path()
    studies.mkdir(parents=True)
    local = studies / "CASE_001.nii.gz"
    local.write_bytes(b"x")

    backend.open_prediction("CASE_001", tmp_path / "pred.nrrd")
    assert launched[0][1] == local
    assert not (settings.results_path() / "_images").exists()


def test_open_prediction_explains_missing_image(backend, settings, mock_server, tmp_path):
    settings.slicer_binary = str(tmp_path / "Slicer.exe")
    mock_server.stop()
    with pytest.raises(RuntimeError, match="Set the Studies folder"):
        backend.open_prediction("CASE_001", tmp_path / "pred.nrrd")
