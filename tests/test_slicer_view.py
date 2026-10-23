from pathlib import PureWindowsPath

from unet_inference_client import slicer_view
from unet_inference_client.backend import MonaiLabelBackend
from unet_inference_client.config import Settings


def fake_installs(monkeypatch, tmp_path, *folders):
    for folder in folders:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "Slicer.exe").write_bytes(b"")
    monkeypatch.setattr(slicer_view, "_install_globs", lambda: [(tmp_path, "*Slicer*/Slicer.exe")])
    monkeypatch.setattr(slicer_view.shutil, "which", lambda name: None)


def test_find_slicer_picks_newest_version(monkeypatch, tmp_path):
    fake_installs(monkeypatch, tmp_path, "Slicer 5.8.1", "3D Slicer 5.12.3", "Slicer 5.6.2")
    assert slicer_view.find_slicer() == tmp_path / "3D Slicer 5.12.3" / "Slicer.exe"


def test_find_slicer_returns_none_when_missing(monkeypatch, tmp_path):
    fake_installs(monkeypatch, tmp_path)
    assert slicer_view.find_slicer() is None


def test_script_uses_forward_slashes_and_is_valid_python():
    image = PureWindowsPath(r"C:\Users\lab\it's here\CASE_1.nrrd")
    prediction = PureWindowsPath(r"C:\results\CASE_1\pred.nrrd")
    script = slicer_view.build_script(image, prediction, "CASE_1")
    assert "C:/Users/lab/it's here/CASE_1.nrrd" in script
    assert "\\" not in script
    compile(script, "view.py", "exec")


def test_open_prediction_uses_detected_slicer(monkeypatch, tmp_path):
    studies = tmp_path / "studies"
    studies.mkdir()
    (studies / "CASE_1.nrrd").write_bytes(b"")
    detected = tmp_path / "Slicer.exe"
    launched = []
    monkeypatch.setattr(slicer_view, "find_slicer", lambda: detected)
    monkeypatch.setattr(slicer_view, "launch", lambda *args: launched.append(args))

    backend = MonaiLabelBackend(Settings(studies_dir=str(studies)))
    backend.open_prediction("CASE_1", tmp_path / "pred.nrrd")
    assert launched[0][0] == detected
