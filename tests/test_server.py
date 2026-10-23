import subprocess
import sys

from unet_inference_client import server

IGNORED_STOP_SIGNAL = "SIGBREAK" if sys.platform == "win32" else "SIGTERM"


def start_sleeper(ignore_stop=False):
    code = "import time; time.sleep(60)"
    if ignore_stop:
        code = (f"import signal, time; signal.signal(signal.{IGNORED_STOP_SIGNAL}, signal.SIG_IGN);"
                " print('ready', flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                             **server.new_process_group())
    if ignore_stop:
        child.stdout.readline()
    return child


def test_stop_ends_server():
    child = start_sleeper()
    process = server.ServerProcess()
    process.process = child
    process.stop(timeout=10)
    assert child.wait(timeout=10) is not None
    assert process.process is None


def test_stop_kills_server_that_ignores_the_stop_signal():
    child = start_sleeper(ignore_stop=True)
    process = server.ServerProcess()
    process.process = child
    process.stop(timeout=1)
    assert child.wait(timeout=10) is not None


def test_working_dir_uses_source_checkout(tmp_path):
    (tmp_path / "monailabel").mkdir()
    app_dir = tmp_path / "sample-apps" / "radiology"
    app_dir.mkdir(parents=True)
    assert server.working_dir(app_dir) == tmp_path


def test_working_dir_falls_back_to_app_folder(tmp_path):
    app_dir = tmp_path / "apps" / "radiology"
    app_dir.mkdir(parents=True)
    assert server.working_dir(app_dir) == app_dir
