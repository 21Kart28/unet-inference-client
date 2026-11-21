import pytest
from PySide6.QtCore import QDeadlineTimer, QEventLoop
from PySide6.QtWidgets import QApplication

from unet_inference_client import app as app_module
from unet_inference_client.app import MainWindow, describe_snapshot, server_problem_hint
from unet_inference_client.backend import ServerSnapshot


@pytest.fixture
def window(qt_app, settings):
    window = MainWindow(settings)
    yield window
    window.close()


def wait_until(condition, timeout_ms=5000):
    deadline = QDeadlineTimer(timeout_ms)
    while not condition():
        if deadline.hasExpired():
            raise TimeoutError("condition not met")
        QApplication.processEvents(QEventLoop.AllEvents, 50)


def test_refresh_runs_off_the_main_thread_and_loads_cases(window):
    assert window.status_label.text() == f"Server: {app_module.CHECKING}"
    assert not window.refresh_button.isEnabled()
    assert not window.run_button.isEnabled()
    assert window.run_button.toolTip() == "Checking the server..."

    wait_until(lambda: window.status_thread is None)

    assert window.status_label.text() == "Server: running on cuda:0"
    assert window.model_label.text() == "Checkpoint epoch 150, Dice 0.9123  \u00b7  3 cases, 2 with labels"
    assert window.case_box.count() == 3
    assert window.case_box.itemText(1) == "CASE_002  (no ground truth)"
    assert window.batch_list.count() == 3
    assert window.run_button.isEnabled()
    assert window.refresh_button.isEnabled()


def test_refresh_reports_unreachable_server(window, mock_server):
    wait_until(lambda: window.status_thread is None)
    mock_server.stop()
    window.refresh()
    wait_until(lambda: window.status_thread is None)

    assert window.status_label.text().startswith("Server: unreachable")
    assert window.model_label.text() == ""
    assert window.case_box.count() == 0
    assert not window.run_button.isEnabled()
    assert window.run_button.toolTip().startswith("Server unreachable")


def test_refresh_ignores_a_second_click_while_checking(window):
    first = window.status_thread
    window.refresh()
    assert window.status_thread is first
    wait_until(lambda: window.status_thread is None)


def test_describe_snapshot_without_training_stats():
    snapshot = ServerSnapshot(True, "running on cpu", [], total_images=5, labeled_images=1)
    assert describe_snapshot(snapshot) == "No training stats  \u00b7  5 cases, 1 with labels"
    assert describe_snapshot(ServerSnapshot(False, "unreachable (x)", [])) == ""


def test_server_problem_hint_while_checking():
    assert server_problem_hint(app_module.CHECKING) == "Checking the server..."


def test_first_run_opens_on_the_setup_tab(qt_app, settings):
    window = MainWindow(settings, first_run=True)
    try:
        assert window.tabs.tabText(window.tabs.currentIndex()) == "Setup"
        wait_until(lambda: window.setup_tab.worker_thread is None and window.status_thread is None)
        names = [check.name for check in window.setup_tab.checks]
        assert names == ["MONAI Label server", "Inference model", "3D Slicer", "Results folder", "Studies folder"]
        assert window.setup_tab.grid.count() == 15
        assert window.setup_tab.headline.text() == "Fix the red items before running inference."
        assert "Setup check: fix Studies folder." in window.log.toPlainText()
    finally:
        window.close()


def test_later_runs_open_on_the_run_tab(window):
    assert window.tabs.tabText(window.tabs.currentIndex()) == "Run"
    wait_until(lambda: window.setup_tab.worker_thread is None and window.status_thread is None)
