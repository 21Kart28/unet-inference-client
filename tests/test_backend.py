from pathlib import Path

import pytest

from unet_inference_client.backend import MonaiLabelBackend
from unet_inference_client.config import Settings


def make_backend(tmp_path):
    output_dir = tmp_path / "CASE_1_run"
    output_dir.mkdir()
    return MonaiLabelBackend(Settings(results_dir=str(tmp_path))), output_dir


def test_latest_falls_back_to_text_pointer(monkeypatch, tmp_path):
    backend, output_dir = make_backend(tmp_path)

    def refuse(*args, **kwargs):
        raise OSError("A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    backend._update_latest_link(output_dir)
    assert (tmp_path / "latest.txt").read_text().strip() == str(output_dir)


def test_latest_symlink_replaces_text_pointer(tmp_path):
    backend, output_dir = make_backend(tmp_path)
    (tmp_path / "latest.txt").write_text("old\n")
    try:
        (tmp_path / "probe").symlink_to(output_dir, target_is_directory=True)
    except OSError:
        pytest.skip("this machine does not allow symlinks")
    backend._update_latest_link(output_dir)
    assert (tmp_path / "latest").resolve() == output_dir.resolve()
    assert not (tmp_path / "latest.txt").exists()
