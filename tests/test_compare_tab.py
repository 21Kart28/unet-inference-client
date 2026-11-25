import numpy as np
import pytest
import SimpleITK as sitk
from PySide6.QtCore import QDeadlineTimer, QEventLoop
from PySide6.QtWidgets import QApplication

from unet_inference_client import theme
from unet_inference_client.compare_tab import CompareTab


@pytest.fixture
def tab(qt_app, settings):
    return CompareTab(settings, lambda text: None)


def test_label_error_clears_when_the_field_changes(tab):
    tab.label_edit.setText("abc")
    tab.compute()
    assert tab.message_label.text().startswith("Label value must be")
    assert theme.RED in tab.message_label.styleSheet()

    tab.label_edit.setText("2")
    assert tab.message_label.text() == ""
    assert not tab.message_is_error


def test_missing_prediction_error_clears_when_a_path_is_typed(tab):
    tab.compute()
    assert tab.message_label.text() == "Choose a prediction file first."

    tab.prediction_edit.setText("C:/somewhere/pred.nrrd")
    assert tab.message_label.text() == ""


def test_per_class_breakdown_fills_the_class_table(tab, tmp_path):
    def write(name, values):
        array = np.zeros((6, 6, 6), dtype=np.uint8)
        for (z, y, x), value in values.items():
            array[z, y, x] = value
        path = tmp_path / f"{name}.nrrd"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        return path

    pred = write("pred", {(1, 1, 1): 1, (2, 2, 2): 2})
    gt = write("gt", {(1, 1, 1): 1, (3, 3, 3): 2})
    tab.prediction_edit.setText(str(pred))
    tab.ground_truth_edit.setText(str(gt))
    tab.hd95_box.setChecked(False)
    tab.per_class_box.setChecked(True)
    tab.compute()

    deadline = QDeadlineTimer(5000)
    while tab.busy:
        assert not deadline.hasExpired(), "compare did not finish"
        QApplication.processEvents(QEventLoop.AllEvents, 50)

    assert tab.class_table.isVisibleTo(tab)
    assert tab.class_table.rowCount() == 2
    assert [tab.class_table.item(r, 0).text() for r in range(2)] == ["1", "2"]
    assert [tab.class_table.item(r, 1).text() for r in range(2)] == ["1.0000", "0.0000"]
    assert tab.value_labels[0][2].text() == "0.5000"
