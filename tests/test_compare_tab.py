import pytest

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
