import numpy as np
import pytest
import SimpleITK as sitk
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from unet_inference_client import dicom_import


def write_series(folder, slices=4, description="Chest CT", patient="PAT 01", uid="1.2.3.4"):
    folder.mkdir(parents=True, exist_ok=True)
    volume = sitk.GetImageFromArray(np.arange(slices * 6 * 5, dtype=np.int16).reshape(slices, 6, 5))
    volume.SetSpacing((0.8, 0.8, 2.5))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for index in range(slices):
        image = volume[:, :, index]
        image.SetMetaData("0020|000e", uid)
        image.SetMetaData("0008|103e", description)
        image.SetMetaData("0010|0020", patient)
        image.SetMetaData("0020|0013", str(index + 1))
        position = volume.TransformIndexToPhysicalPoint((0, 0, index))
        image.SetMetaData("0020|0032", "\\".join(f"{v:.3f}" for v in position))
        image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
        writer.SetFileName(str(folder / f"slice_{index:03d}.dcm"))
        writer.Execute(image)
    return folder


def test_find_series_reads_description_and_slice_count(tmp_path):
    write_series(tmp_path / "scan")
    (series,) = dicom_import.find_series(tmp_path / "scan")
    assert series.description == "Chest CT"
    assert len(series.files) == 4
    assert series.label == "Chest CT (4 slices)"


def test_folder_without_dicom_explains_what_to_pick(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("hello")
    with pytest.raises(dicom_import.NoSeriesFound, match="No DICOM series found in empty. Pick the folder"):
        dicom_import.find_series(tmp_path / "empty")
    with pytest.raises(dicom_import.NoSeriesFound, match="is not a folder"):
        dicom_import.find_series(tmp_path / "missing")


def test_case_id_prefers_patient_id_and_is_filename_safe(tmp_path):
    write_series(tmp_path / "scan dir")
    (series,) = dicom_import.find_series(tmp_path / "scan dir")
    assert dicom_import.case_id_for(tmp_path / "scan dir", series) == "PAT_01"
    assert dicom_import.case_id_for(tmp_path / "scan dir") == "scan_dir"


def test_convert_writes_a_volume_with_the_series_geometry(tmp_path):
    write_series(tmp_path / "scan")
    (series,) = dicom_import.find_series(tmp_path / "scan")
    out = dicom_import.convert(series, tmp_path / "studies" / "case.nrrd")
    image = sitk.ReadImage(str(out))
    assert image.GetSize() == (5, 6, 4)
    assert image.GetSpacing() == pytest.approx((0.8, 0.8, 2.5))


def run_importer(qt_app, folder, destination):
    importer = dicom_import.DicomImporter()
    events = []
    importer.imported.connect(lambda path: events.append(("ok", path)))
    importer.failed.connect(lambda message: events.append(("failed", message)))
    importer.start(folder, destination)
    loop = QEventLoop()
    QTimer.singleShot(10000, loop.quit)
    importer.imported.connect(loop.quit)
    importer.failed.connect(loop.quit)
    loop.exec()
    if importer.worker_thread is not None:
        importer.worker_thread.wait(5000)
    QCoreApplication.processEvents()
    return importer, events


def test_importer_converts_on_a_thread_and_names_the_file_after_the_case(qt_app, tmp_path):
    write_series(tmp_path / "scan")
    importer, events = run_importer(qt_app, tmp_path / "scan", tmp_path / "studies")
    assert events == [("ok", tmp_path / "studies" / "PAT_01.nrrd")]
    assert events[0][1].is_file()
    assert not importer.busy


def test_importer_reports_a_folder_without_series(qt_app, tmp_path):
    (tmp_path / "empty").mkdir()
    _, events = run_importer(qt_app, tmp_path / "empty", tmp_path / "studies")
    assert events[0][0] == "failed"
    assert events[0][1].startswith("No DICOM series found in empty.")
