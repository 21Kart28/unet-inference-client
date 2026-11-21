"""Checks that a fresh install can run, each with a hint for fixing it."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import requests

from unet_inference_client import slicer_view
from unet_inference_client.backend import MonaiLabelBackend
from unet_inference_client.config import Settings

OK = "ok"
PROBLEM = "problem"
OPTIONAL = "optional"


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str
    hint: str = ""


def run_checks(settings: Settings, backend: MonaiLabelBackend) -> list[Check]:
    checks = list(_server_checks(settings, backend))
    checks.append(_slicer_check(settings))
    checks.append(_results_check(settings))
    checks.append(_studies_check(settings))
    return checks


def _server_checks(settings: Settings, backend: MonaiLabelBackend) -> list[Check]:
    try:
        info = backend.fetch_info()
    except requests.RequestException as exc:
        return [
            Check("MONAI Label server", PROBLEM, f"{settings.server_url} did not answer ({type(exc).__name__}).",
                  "Check the Server URL in Settings, or start the server on the Server tab."),
            Check("Inference model", PROBLEM, "Can't check until the server answers."),
        ]
    models = info.get("models", {})
    server = Check("MONAI Label server", OK, f"Answering at {settings.server_url}.")
    if settings.model in models:
        device = models[settings.model].get("config", {}).get("device", ["?"])[0]
        return [server, Check("Inference model", OK, f"'{settings.model}' is loaded on {device}.")]
    available = ", ".join(models) or "none"
    return [server, Check("Inference model", PROBLEM, f"'{settings.model}' is not loaded (available: {available}).",
                          "Set Inference model in Settings to one of the available names.")]


def _slicer_check(settings: Settings) -> Check:
    chosen = settings.slicer_path()
    if chosen is not None:
        if chosen.exists():
            return Check("3D Slicer", OK, f"Using {chosen}.")
        return Check("3D Slicer", PROBLEM, f"{chosen} does not exist.",
                     "Fix the Slicer executable in Settings, or clear it to auto-detect.")
    found = slicer_view.find_slicer()
    if found is not None:
        return Check("3D Slicer", OK, f"Found {found}.")
    return Check("3D Slicer", OPTIONAL, "Not found. Inference works, but View in Slicer won't.",
                 "Install 3D Slicer from slicer.org, or set the Slicer executable in Settings.")


def _results_check(settings: Settings) -> Check:
    folder = settings.results_path()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder):
            pass
    except OSError as exc:
        return Check("Results folder", PROBLEM, f"Can't write to {folder} ({exc.strerror or exc}).",
                     "Pick a folder you can write to under Results folder in Settings.")
    return Check("Results folder", OK, f"Writable: {folder}.")


def _studies_check(settings: Settings) -> Check:
    folder = settings.studies_path()
    if folder is None:
        return Check("Studies folder", OPTIONAL, "Not set. Source images will be downloaded from the server.",
                     "Set Studies folder in Settings if the scans are on this machine.")
    if not folder.is_dir():
        return Check("Studies folder", PROBLEM, f"{folder} is not a folder.",
                     "Fix Studies folder in Settings, or clear it to download images instead.")
    labels = folder / "labels" / "final"
    count = sum(1 for entry in os.scandir(labels) if entry.is_file()) if labels.is_dir() else 0
    return Check("Studies folder", OK, f"{folder} with {count} ground-truth label(s).")
