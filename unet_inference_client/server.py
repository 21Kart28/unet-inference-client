"""Starts, stops, and monitors a MONAI Label server process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import requests


def build_command(python_binary: Path, app_dir: Path, studies_dir: Path,
                  models: str, port: int) -> list[str]:
    return [
        str(python_binary), "-m", "monailabel.main", "start_server",
        "--app", str(app_dir),
        "--studies", str(studies_dir),
        "--conf", "models", models,
        "--port", str(port),
    ]


def working_dir(app_dir: Path) -> Path:
    # In a MONAI Label source checkout the app sits at <repo>/<apps>/<app>, and running from
    # <repo> lets `python -m monailabel.main` import that checkout. Pip installs don't need it.
    checkout = app_dir.parent.parent
    return checkout if (checkout / "monailabel").is_dir() else app_dir


def new_process_group() -> dict:
    # Keeps Ctrl+C in the app's console away from the server, and lets stop() signal it alone.
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def is_responding(server_url: str, timeout: float = 2.0) -> bool:
    try:
        requests.get(f"{server_url.rstrip('/')}/info/", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def find_external(port: int) -> list[int]:
    """PIDs of monailabel servers this app did not start."""
    try:
        output = subprocess.run(
            ["pgrep", "-f", "monailabel.main start_server"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in output.split() if line.isdigit()]


class ServerProcess:
    """Owns a server subprocess and exposes its output line by line."""

    def __init__(self):
        self.process = None
        self.command = []

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, python_binary: Path, app_dir: Path, studies_dir: Path,
              models: str, port: int, gpu_index: str) -> None:
        if self.running:
            raise RuntimeError("This app already started a server.")
        if not python_binary.exists():
            raise RuntimeError(f"Python not found at {python_binary}")
        if not app_dir.is_dir():
            raise RuntimeError(f"App folder not found at {app_dir}")
        if not studies_dir.is_dir():
            raise RuntimeError(f"Studies folder not found at {studies_dir}")

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_index
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"

        self.command = build_command(python_binary, app_dir, studies_dir, models, port)
        self.process = subprocess.Popen(
            self.command,
            cwd=str(working_dir(app_dir)),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **new_process_group(),
        )

    def read_line(self) -> str | None:
        """Returns the next output line, or None when there is nothing to read."""
        if self.process is None or self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        return line.rstrip() if line else None

    def stop(self, timeout: float = 15.0) -> None:
        if not self.running:
            self.process = None
            return
        try:
            self._ask_to_stop()
            self.process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            self._kill()
        self.process = None

    def _ask_to_stop(self) -> None:
        if sys.platform == "win32":
            # Windows has no SIGTERM for console programs; uvicorn shuts down cleanly on Ctrl+Break.
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    def _kill(self) -> None:
        if self.process.poll() is not None:
            return
        if sys.platform == "win32":
            # /T also ends any worker processes the server started.
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

    def detach(self) -> None:
        """Forget the process without killing it, so it survives app exit."""
        self.process = None
