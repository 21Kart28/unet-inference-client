"""Setup tab: runs the install checks on a thread and shows each as green, amber or red."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from unet_inference_client import logs, setup_check, theme
from unet_inference_client.backend import MonaiLabelBackend
from unet_inference_client.config import Settings

COLORS = {setup_check.OK: theme.GREEN, setup_check.OPTIONAL: theme.AMBER, setup_check.PROBLEM: theme.RED}
HEADLINES = {
    setup_check.OK: "Everything is ready.",
    setup_check.OPTIONAL: "Ready to run. The amber items are optional.",
    setup_check.PROBLEM: "Fix the red items before running inference.",
}


class CheckWorker(QObject):
    finished = Signal(object)

    def __init__(self, settings: Settings, backend: MonaiLabelBackend):
        super().__init__()
        self.settings = settings
        self.backend = backend

    def run(self) -> None:
        self.finished.emit(setup_check.run_checks(self.settings, self.backend))


class SetupTab(QWidget):
    def __init__(self, settings: Settings, backend: MonaiLabelBackend, log: Callable[[str], None]):
        super().__init__()
        self.settings = settings
        self.backend = backend
        self.log = log
        self.worker_thread = None
        self.worker = None
        self.checks = []

        self.headline = QLabel("Checking this computer...")
        self.headline.setStyleSheet("font-size: 15px;")
        self.rerun_button = QPushButton("Check again")
        self.rerun_button.clicked.connect(self.run)
        self.log_button = QPushButton("Open log folder")
        self.log_button.clicked.connect(self.open_log_folder)

        top = QHBoxLayout()
        top.addWidget(self.headline, stretch=1)
        top.addWidget(self.log_button)
        top.addWidget(self.rerun_button)

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)
        self.grid.setColumnStretch(2, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(self.grid)
        layout.addStretch()

    def run(self) -> None:
        if self.worker_thread is not None:
            return
        self.headline.setText("Checking this computer...")
        self.rerun_button.setEnabled(False)
        self.worker_thread = QThread()
        self.worker = CheckWorker(self.settings, self.backend)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._show)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._thread_done)
        self.worker_thread.start()

    def _thread_done(self) -> None:
        self.worker_thread = None
        self.worker = None
        self.rerun_button.setEnabled(True)

    def _show(self, checks: list[setup_check.Check]) -> None:
        self.checks = checks
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        for row, check in enumerate(checks):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {COLORS[check.state]}; font-size: 16px;")
            dot.setAlignment(Qt.AlignmentFlag.AlignTop)
            name = QLabel(check.name)
            name.setAlignment(Qt.AlignmentFlag.AlignTop)
            text = check.detail if not check.hint else f"{check.detail}\n{check.hint}"
            detail = QLabel(text)
            detail.setWordWrap(True)
            detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            detail.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self.grid.addWidget(dot, row, 0)
            self.grid.addWidget(name, row, 1)
            self.grid.addWidget(detail, row, 2)
        worst = setup_check.OK
        for check in checks:
            if check.state == setup_check.PROBLEM:
                worst = setup_check.PROBLEM
            elif check.state == setup_check.OPTIONAL and worst == setup_check.OK:
                worst = setup_check.OPTIONAL
        self.headline.setText(HEADLINES[worst])
        self.headline.setStyleSheet(f"font-size: 15px; color: {COLORS[worst]};")
        problems = [check.name for check in checks if check.state == setup_check.PROBLEM]
        if problems:
            self.log(f"Setup check: fix {', '.join(problems)}.")

    def open_log_folder(self) -> None:
        if logs.configure() is None:
            self.log(f"Could not create {logs.LOG_DIR}. Check that the folder is writable.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs.LOG_DIR))):
            self.log(f"Could not open {logs.LOG_DIR}. Open it in the file manager instead.")

    def close_thread(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
