from importlib.metadata import entry_points
from pathlib import Path

from unet_inference_client import app


def test_unet_client_command_runs_main():
    (script,) = entry_points(group="console_scripts", name="unet-client")
    assert script.load() is app.main


def test_pyinstaller_spec_points_at_the_entry_module():
    spec = Path(__file__).resolve().parents[1] / "unet-client.spec"
    text = spec.read_text(encoding="utf-8")
    compile(text, str(spec), "exec")
    assert 'unet_inference_client/__main__.py' in text
    assert 'name="unet-client"' in text
