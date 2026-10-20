from importlib.metadata import entry_points

from unet_inference_client import app


def test_unet_client_command_runs_main():
    (script,) = entry_points(group="console_scripts", name="unet-client")
    assert script.load() is app.main
