"""Convert a DICOM series folder to an NRRD volume the MONAI Label server can serve."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk
from PySide6.QtCore import QObject, QThread, Signal

SERIES_DESCRIPTION_TAG = "0008|103e"
PATIENT_ID_TAG = "0010|0020"


class NoSeriesFound(ValueError):
    """The folder holds no readable DICOM series."""


@dataclass(frozen=True)
class Series:
    uid: str
    description: str
    files: list[str]

    @property
    def label(self) -> str:
        name = self.description or self.uid[-12:]
        return f"{name} ({len(self.files)} slices)"


def find_series(folder: Path | str) -> list[Series]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NoSeriesFound(f"{folder} is not a folder.")
    reader = sitk.ImageSeriesReader()
    found = []
    for uid in reader.GetGDCMSeriesIDs(str(folder)):
        files = list(reader.GetGDCMSeriesFileNames(str(folder), uid))
        if not files:
            continue
        found.append(Series(uid, _read_tag(files[0], SERIES_DESCRIPTION_TAG), files))
    if not found:
        raise NoSeriesFound(
            f"No DICOM series found in {folder.name}. Pick the folder that holds the .dcm slices of one scan."
        )
    return sorted(found, key=lambda series: len(series.files), reverse=True)


def case_id_for(folder: Path | str, series: Series | None = None) -> str:
    """A file-system safe case name: the patient ID if the slices carry one, else the folder name."""
    patient = _read_tag(series.files[0], PATIENT_ID_TAG) if series else ""
    raw = patient or Path(folder).name or "case"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "case"


def convert(series: Series, destination: Path | str) -> Path:
    """Read every slice of `series` and write it as one volume at `destination`."""
    destination = Path(destination)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series.files)
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise ValueError(f"Could not read the DICOM slices: {_first_line(exc)}") from None
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(destination), useCompression=True)
    return destination


class ImportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path, destination_dir: Path):
        super().__init__()
        self.folder = folder
        self.destination_dir = destination_dir

    def run(self) -> None:
        try:
            series = find_series(self.folder)[0]
            case_id = case_id_for(self.folder, series)
            path = convert(series, self.destination_dir / f"{case_id}.nrrd")
        except (NoSeriesFound, ValueError, OSError) as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(path)


class DicomImporter(QObject):
    """Runs one folder import at a time on its own thread and relays the outcome."""

    imported = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.worker_thread: QThread | None = None
        self.worker: ImportWorker | None = None

    @property
    def busy(self) -> bool:
        return self.worker_thread is not None

    def start(self, folder: Path, destination_dir: Path) -> None:
        if self.busy:
            self.failed.emit("An import is already running. Wait for it to finish.")
            return
        self.worker_thread = QThread()
        self.worker = ImportWorker(folder, destination_dir)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.imported)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._thread_done)
        self.worker_thread.start()

    def _thread_done(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None

    def close(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(5000)


def _read_tag(path: str, tag: str) -> str:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOff()
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return ""
    return reader.GetMetaData(tag).strip() if reader.HasMetaDataKey(tag) else ""


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[-1] if str(exc).strip() else "unknown error"
