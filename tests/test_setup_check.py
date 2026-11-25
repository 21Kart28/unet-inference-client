import pytest

from unet_inference_client import slicer_view
from unet_inference_client.setup_check import OK, OPTIONAL, PROBLEM, run_checks


def by_name(checks):
    return {check.name: check for check in checks}


@pytest.fixture
def slicer(tmp_path, settings):
    binary = tmp_path / "Slicer.exe"
    binary.write_bytes(b"")
    settings.slicer_binary = str(binary)
    return binary


def test_everything_green_with_mock_server_and_slicer(settings, backend, slicer):
    settings.studies_path().joinpath("labels", "final").mkdir(parents=True)
    settings.studies_path().joinpath("labels", "final", "CASE_001.nrrd").write_bytes(b"")
    checks = by_name(run_checks(settings, backend))
    assert [c.state for c in checks.values()] == [OK] * 5
    assert checks["Inference model"].detail == "'deepedit_seg' is loaded on cuda:0."
    assert checks["Studies folder"].detail.endswith("with 1 ground-truth label(s).")


def test_unreachable_server_blocks_model_check(settings, backend, mock_server):
    mock_server.stop()
    checks = by_name(run_checks(settings, backend))
    assert checks["MONAI Label server"].state == PROBLEM
    assert "ConnectionError" in checks["MONAI Label server"].detail
    assert checks["MONAI Label server"].hint.startswith("Check the Server URL")
    assert checks["Inference model"].state == PROBLEM


def test_model_not_loaded_lists_available(settings, backend, mock_server):
    mock_server.models = ["spleen"]
    check = by_name(run_checks(settings, backend))["Inference model"]
    assert check.state == PROBLEM
    assert "available: spleen" in check.detail
    assert "Settings" in check.hint


def test_slicer_missing_is_optional(settings, backend, monkeypatch):
    monkeypatch.setattr(slicer_view, "find_slicer", lambda: None)
    settings.studies_dir = ""
    check = by_name(run_checks(settings, backend))["3D Slicer"]
    assert check.state == OPTIONAL
    assert "slicer.org" in check.hint


def test_slicer_wrong_path_is_a_problem(settings, backend, tmp_path):
    settings.slicer_binary = str(tmp_path / "nope" / "Slicer.exe")
    check = by_name(run_checks(settings, backend))["3D Slicer"]
    assert check.state == PROBLEM
    assert "clear it to auto-detect" in check.hint


def test_unwritable_results_folder(settings, backend, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    settings.results_dir = str(blocker / "results")
    check = by_name(run_checks(settings, backend))["Results folder"]
    assert check.state == PROBLEM
    assert "Pick a folder you can write to" in check.hint


def test_studies_folder_unset_is_optional(settings, backend):
    settings.studies_dir = ""
    check = by_name(run_checks(settings, backend))["Studies folder"]
    assert check.state == OPTIONAL
    assert "downloaded from the server" in check.detail


def test_studies_folder_missing_is_a_problem(settings, backend):
    check = by_name(run_checks(settings, backend))["Studies folder"]
    assert check.state == PROBLEM
    assert "is not a folder" in check.detail
