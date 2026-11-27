"""Main window for U-Net Inference Client."""

import logging
import shutil
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from unet_inference_client import batch, history, logs, metrics, server, slicer_view, theme
from unet_inference_client.backend import MonaiLabelBackend, ServerSnapshot
from unet_inference_client.compare_tab import CompareTab, format_dice, format_volume
from unet_inference_client.config import CONFIG_FILE, Settings
from unet_inference_client.dicom_import import DicomImporter
from unet_inference_client.model_compare_tab import ModelCompareTab
from unet_inference_client.setup_tab import SetupTab

CHECKING = "checking..."


ui_log = logging.getLogger("unet_inference_client.ui")


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def describe_snapshot(snapshot: ServerSnapshot) -> str:
    if not snapshot.is_up:
        return ""
    checkpoint = f"Checkpoint epoch {snapshot.epoch}, Dice {snapshot.dice}"
    if snapshot.epoch == "unknown":
        checkpoint = "No training stats"
    return f"{checkpoint}  ·  {snapshot.total_images} cases, {snapshot.labeled_images} with labels"


def server_problem_hint(message: str) -> str:
    if message == CHECKING:
        return "Checking the server..."
    if message.startswith("unreachable"):
        return "Server unreachable. Check the Server URL in Settings, or start the server on the Server tab."
    return f"Server {message}. Check the inference model in Settings."


class StatusWorker(QObject):
    finished = Signal(object)

    def __init__(self, backend: MonaiLabelBackend):
        super().__init__()
        self.backend = backend

    def run(self) -> None:
        self.finished.emit(self.backend.snapshot())


class ServerReader(QObject):
    line = Signal(str)
    stopped = Signal()

    def __init__(self, process: server.ServerProcess):
        super().__init__()
        self.process = process

    def run(self) -> None:
        while self.process.running:
            text = self.process.read_line()
            if text is None:
                break
            self.line.emit(text)
        self.stopped.emit()


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, first_run: bool = False):
        super().__init__()
        self.settings = settings
        self.backend = MonaiLabelBackend(settings)
        self.server = server.ServerProcess()
        self.importer = DicomImporter()
        self.importer.imported.connect(self.on_dicom_imported)
        self.importer.failed.connect(self.on_dicom_failed)
        self.thread = None
        self.worker = None
        self.reader_thread = None
        self.reader = None
        self.status_thread = None
        self.status_worker = None
        self.batch_started_at = None
        self.case_started_at = None
        self.stored_results = []
        self.inference_busy = False
        self.server_up = False
        self.server_message = CHECKING

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)

        self.setWindowTitle("U-Net Inference Client")
        self.resize(960, 740)

        self.status_label = QLabel(f"Server: {CHECKING}")
        self.model_label = QLabel("")
        self.model_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.refresh_button = QPushButton("Refresh")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        header = QHBoxLayout()
        header.addWidget(self.status_label)
        header.addSpacing(16)
        header.addWidget(self.model_label)
        header.addStretch()
        header.addWidget(self.refresh_button)

        self.tabs = QTabWidget()
        self.setup_tab = SetupTab(settings, self.backend, self.append)
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_batch_tab(), "Batch")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.compare_tab = CompareTab(settings, self.append)
        self.compare_tab.metrics_saved.connect(self.reload_results)
        self.tabs.addTab(self.compare_tab, "Compare")
        self.model_compare_tab = ModelCompareTab(settings, self.append)
        self.tabs.addTab(self.model_compare_tab, "Models")
        self.tabs.addTab(self._build_server_tab(), "Server")
        self.tabs.addTab(self._build_settings_tab(), "Settings")

        log_label = QLabel("Log")
        log_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.tabs, stretch=3)
        layout.addWidget(self._build_status_strip())
        layout.addWidget(log_label)
        layout.addWidget(self.log, stretch=2)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.refresh_button.clicked.connect(self.refresh)
        self.refresh()
        self.reload_results()
        self.setup_tab.run()
        self.tabs.setCurrentIndex(0 if first_run else 1)

    def _build_status_strip(self) -> QWidget:
        self.activity_label = QLabel("Idle")
        self.activity_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.case_timer_label = QLabel("")
        self.case_timer_label.setStyleSheet(
            f"color: {theme.AMBER}; font-family: monospace; font-size: 15px;"
        )
        self.total_timer_label = QLabel("")
        self.total_timer_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-family: monospace;")
        self.batch_progress = QProgressBar()
        self.batch_progress.setTextVisible(False)
        self.batch_progress.setMaximum(1)
        self.batch_progress.setValue(0)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        top = QHBoxLayout()
        top.addWidget(self.activity_label, stretch=1)
        top.addWidget(self.case_timer_label)
        top.addWidget(self.total_timer_label)

        bottom = QHBoxLayout()
        bottom.addWidget(self.batch_progress, stretch=1)
        bottom.addWidget(self.count_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addLayout(bottom)

        strip = QFrame()
        strip.setObjectName("statusStrip")
        strip.setLayout(layout)
        strip.setStyleSheet(
            f"#statusStrip {{ background-color: {theme.SURFACE};"
            f" border: 1px solid {theme.BORDER}; border-radius: 6px; }}"
            " #statusStrip QLabel { background-color: transparent; }"
        )
        return strip

    def _build_run_tab(self) -> QWidget:
        self.case_box = QComboBox()
        self.case_box.setMaxVisibleItems(12)
        self.run_button = QPushButton("Run inference")
        self.slicer_button = QPushButton("View in Slicer")
        self.import_button = QPushButton("Import DICOM folder")
        self.run_button.clicked.connect(self.start_single)
        self.slicer_button.clicked.connect(self.open_slicer)
        self.import_button.clicked.connect(self.import_dicom)

        row = QHBoxLayout()
        row.addWidget(QLabel("Case"))
        row.addWidget(self.case_box, stretch=1)
        row.addWidget(self.run_button)
        row.addWidget(self.slicer_button)
        row.addWidget(self.import_button)

        hint = QLabel("Runs the model on one case. View opens an existing prediction in 3D Slicer."
                      " Import converts a DICOM series to NRRD in the studies folder.")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM};")

        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addWidget(hint)
        layout.addStretch()
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_batch_tab(self) -> QWidget:
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.skip_existing_box = QCheckBox("Skip cases that already have a result")
        self.skip_existing_box.setChecked(self.settings.skip_existing)
        self.select_all_button = QPushButton("Select all")
        self.select_none_button = QPushButton("Select none")
        self.batch_run_button = QPushButton("Run batch")
        self.batch_resume_button = QPushButton("Resume")
        self.batch_resume_button.setEnabled(False)
        self.batch_resume_button.setToolTip("Continue a cancelled batch from where it stopped.")
        self.batch_cancel_button = QPushButton("Cancel")
        self.batch_cancel_button.setEnabled(False)
        self.remaining_cases = []

        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        self.batch_run_button.clicked.connect(self.start_batch)
        self.batch_resume_button.clicked.connect(self.resume_batch)
        self.batch_cancel_button.clicked.connect(self.cancel_batch)

        buttons = QHBoxLayout()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.select_none_button)
        buttons.addStretch()
        buttons.addWidget(self.batch_run_button)
        buttons.addWidget(self.batch_resume_button)
        buttons.addWidget(self.batch_cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(self.batch_list, stretch=1)
        layout.addWidget(self.skip_existing_box)
        layout.addLayout(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_results_tab(self) -> QWidget:
        headers = ["Case", "Model", "Epoch", "Val Dice", "Volume (mm³)", "Dice vs GT", "Saved"]
        self.results_table = QTableWidget(0, len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeaderItem(3).setToolTip("Validation Dice of the checkpoint")
        self.results_table.horizontalHeaderItem(5).setToolTip("Dice against ground truth, from the Compare tab")
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.cellDoubleClicked.connect(self.view_selected_result)

        self.results_count_label = QLabel("")
        self.results_count_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        reload_button = QPushButton("Reload")
        view_button = QPushButton("View in Slicer")
        compare_button = QPushButton("Compare")
        export_button = QPushButton("Export history CSV")
        reload_button.clicked.connect(self.reload_results)
        view_button.clicked.connect(self.view_selected_result)
        compare_button.clicked.connect(self.compare_selected_result)
        export_button.clicked.connect(self.export_history)

        buttons = QHBoxLayout()
        buttons.addWidget(self.results_count_label)
        buttons.addStretch()
        buttons.addWidget(reload_button)
        buttons.addWidget(view_button)
        buttons.addWidget(compare_button)
        buttons.addWidget(export_button)

        layout = QVBoxLayout()
        layout.addWidget(self.results_table, stretch=1)
        layout.addLayout(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_server_tab(self) -> QWidget:
        self.server_python_edit = QLineEdit(self.settings.server_python)
        self.server_app_edit = QLineEdit(self.settings.server_app_dir)
        self.server_studies_edit = QLineEdit(self.settings.studies_dir)
        self.server_models_edit = QLineEdit(self.settings.server_models)
        self.server_port_edit = QLineEdit(str(self.settings.server_port))
        self.gpu_edit = QLineEdit(self.settings.gpu_index)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("MONAI Label python", self._with_browse(self.server_python_edit, directory=False))
        form.addRow("App folder", self._with_browse(self.server_app_edit, directory=True))
        form.addRow("Studies folder", self._with_browse(self.server_studies_edit, directory=True))
        form.addRow("Models", self.server_models_edit)
        form.addRow("Port", self.server_port_edit)
        form.addRow("GPU index", self.gpu_edit)

        self.server_start_button = QPushButton("Start server")
        self.server_stop_button = QPushButton("Stop server")
        self.server_stop_button.setEnabled(False)
        self.server_start_button.clicked.connect(self.start_server)
        self.server_stop_button.clicked.connect(self.stop_server)

        note = QLabel(
            "Starting a server here changes which studies folder is served. "
            "A server started by this app keeps running after the app closes."
        )
        note.setStyleSheet(f"color: {theme.TEXT_DIM};")
        note.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.server_start_button)
        buttons.addWidget(self.server_stop_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addLayout(buttons)
        layout.addStretch()
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_settings_tab(self) -> QWidget:
        self.server_url_edit = QLineEdit(self.settings.server_url)
        self.model_edit = QLineEdit(self.settings.model)
        self.resolution_edit = QLineEdit(self.settings.resolution)
        self.results_edit = QLineEdit(self.settings.results_dir)
        self.studies_edit = QLineEdit(self.settings.studies_dir)
        self.slicer_edit = QLineEdit(self.settings.slicer_binary)
        detected = slicer_view.find_slicer()
        self.slicer_edit.setPlaceholderText(
            f"Leave blank to use {detected}" if detected else "Couldn't find 3D Slicer. Use Browse to pick it."
        )
        self.model_dir_edit = QLineEdit(self.settings.model_dir)

        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Server URL", self.server_url_edit)
        form.addRow("Inference model", self.model_edit)
        form.addRow("Resolution label", self.resolution_edit)
        form.addRow("Results folder", self._with_browse(self.results_edit, directory=True))
        form.addRow("Studies folder", self._with_browse(self.studies_edit, directory=True))
        form.addRow("Slicer executable", self._with_browse(self.slicer_edit, directory=False))
        form.addRow("Model folder", self._with_browse(self.model_dir_edit, directory=True))

        save_button = QPushButton("Save settings")
        save_button.clicked.connect(self.save_settings)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(save_button)
        layout.addStretch()
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _with_browse(self, line_edit: QLineEdit, directory: bool) -> QWidget:
        button = QPushButton("Browse")

        def browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(self, "Select folder", line_edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text())
            if path:
                line_edit.setText(path)

        button.clicked.connect(browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(button)
        container = QWidget()
        container.setLayout(row)
        return container

    def append(self, text: str) -> None:
        self.log.appendPlainText(text)
        ui_log.info(text)

    def save_settings(self) -> None:
        self.settings.server_url = self.server_url_edit.text().strip()
        self.settings.model = self.model_edit.text().strip()
        self.settings.resolution = self.resolution_edit.text().strip()
        self.settings.results_dir = self.results_edit.text().strip()
        self.settings.studies_dir = self.studies_edit.text().strip()
        self.settings.slicer_binary = self.slicer_edit.text().strip()
        self.settings.model_dir = self.model_dir_edit.text().strip()
        self.settings.skip_existing = self.skip_existing_box.isChecked()
        self.settings.save()
        self.server_studies_edit.setText(self.settings.studies_dir)
        self.append("Settings saved.")
        self.refresh()
        self.reload_results()
        self.setup_tab.run()

    def save_server_settings(self) -> None:
        self.settings.server_python = self.server_python_edit.text().strip()
        self.settings.server_app_dir = self.server_app_edit.text().strip()
        self.settings.studies_dir = self.server_studies_edit.text().strip()
        self.settings.server_models = self.server_models_edit.text().strip()
        try:
            self.settings.server_port = int(self.server_port_edit.text().strip())
        except ValueError:
            raise RuntimeError("Port must be a number.") from None
        self.settings.gpu_index = self.gpu_edit.text().strip()
        self.settings.server_url = f"http://127.0.0.1:{self.settings.server_port}"
        self.settings.save()
        self.server_url_edit.setText(self.settings.server_url)
        self.studies_edit.setText(self.settings.studies_dir)

    def start_server(self) -> None:
        try:
            self.save_server_settings()
        except RuntimeError as exc:
            self.append(str(exc))
            return

        if server.is_responding(self.settings.server_url):
            self.append("A server is already answering on that port. Stop it first.")
            return

        try:
            self.server.start(
                self.settings.server_python_path(),
                self.settings.server_app_path(),
                self.settings.studies_path(),
                self.settings.server_models,
                self.settings.server_port,
                self.settings.gpu_index,
            )
        except Exception as exc:
            self.append(f"Could not start server: {exc}")
            return

        self.append(f"Starting server on port {self.settings.server_port}...")
        self.append(f"  studies: {self.settings.studies_dir}")
        self.server_start_button.setEnabled(False)
        self.server_stop_button.setEnabled(True)

        self.reader_thread = QThread()
        self.reader = ServerReader(self.server)
        self.reader.moveToThread(self.reader_thread)
        self.reader_thread.started.connect(self.reader.run)
        self.reader.line.connect(self.on_server_line)
        self.reader.stopped.connect(self.reader_thread.quit)
        self.reader_thread.finished.connect(self.reader_thread.deleteLater)
        self.reader_thread.start()

    def on_server_line(self, text: str) -> None:
        if "Uvicorn running on" in text or "Application startup complete" in text:
            self.append(f"  {text}")
            QTimer.singleShot(1500, self.refresh)
        elif "ERROR" in text or "Traceback" in text:
            self.append(f"  {text}")

    def stop_server(self) -> None:
        if not self.server.running:
            self.append("This app is not managing a server.")
            self.server_start_button.setEnabled(True)
            self.server_stop_button.setEnabled(False)
            return
        self.append("Stopping server...")
        try:
            self.server.stop()
            self.append("Server stopped.")
        except Exception as exc:
            self.append(f"Could not stop server: {exc}")
        self.server_start_button.setEnabled(True)
        self.server_stop_button.setEnabled(False)
        self.refresh()

    def refresh(self) -> None:
        if self.status_thread is not None:
            return
        self.status_label.setText(f"Server: {CHECKING}")
        self.status_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.refresh_button.setEnabled(False)
        self.server_up, self.server_message = False, CHECKING
        self._update_run_buttons()
        self.status_thread = QThread()
        self.status_worker = StatusWorker(self.backend)
        self.status_worker.moveToThread(self.status_thread)
        self.status_thread.started.connect(self.status_worker.run)
        self.status_worker.finished.connect(self.on_snapshot)
        self.status_worker.finished.connect(self.status_thread.quit)
        self.status_thread.finished.connect(self._status_thread_done)
        self.status_thread.start()

    def _status_thread_done(self) -> None:
        self.status_thread = None
        self.status_worker = None
        self.refresh_button.setEnabled(not self.inference_busy)

    def on_snapshot(self, snapshot: ServerSnapshot) -> None:
        self.status_label.setText(f"Server: {snapshot.message}")
        color = theme.GREEN if snapshot.is_up else theme.RED
        self.status_label.setStyleSheet(f"color: {color};")
        if snapshot.message != self.server_message:
            self.append(f"Server: {snapshot.message}")
        self.server_up, self.server_message = snapshot.is_up, snapshot.message
        self.model_label.setText(describe_snapshot(snapshot))
        self._update_run_buttons()
        self.case_box.clear()
        self.batch_list.clear()
        if not snapshot.is_up:
            return
        if snapshot.cases_error:
            self.append(f"Could not load cases: {snapshot.cases_error}")
            return
        for case in snapshot.cases:
            suffix = "" if case.has_ground_truth else "  (no ground truth)"
            self.case_box.addItem(f"{case.case_id}{suffix}", case.case_id)
            item = QListWidgetItem(case.case_id)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.batch_list.addItem(item)
        self.append(f"Loaded {len(snapshot.cases)} cases.")

    def reload_results(self) -> None:
        stored = history.scan(self.settings.results_path())
        self.stored_results = stored
        self.results_table.setRowCount(len(stored))
        for row, result in enumerate(stored):
            saved = metrics.load_metrics(result.output_dir) or {}
            values = [
                result.case_id, result.model, result.epoch, result.dice,
                format_volume(saved.get("pred_volume_mm3")), format_dice(saved.get("dice")),
                result.timestamp,
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.results_count_label.setText(f"{len(stored)} result(s) in {self.settings.results_dir}")
        self.compare_tab.set_results(stored)
        self.model_compare_tab.set_results(stored)

    def _selected_result(self) -> history.StoredResult | None:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.stored_results):
            self.append("Select a result first.")
            return None
        return self.stored_results[row]

    def view_selected_result(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        try:
            self.backend.open_prediction(result.case_id, result.prediction_path)
            self.append(f"Launching Slicer for {result.case_id}...")
        except Exception as exc:
            self.append(f"Could not launch Slicer: {exc}")

    def compare_selected_result(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        self.compare_tab.load_result(result)
        self.tabs.setCurrentWidget(self.compare_tab)

    def export_history(self) -> None:
        source = history.log_path(self.settings.results_path())
        if not source.exists():
            self.append("No history recorded yet.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export history", str(source.name), "CSV files (*.csv)"
        )
        if not destination:
            return
        shutil.copy(source, destination)
        self.append(f"History exported to {destination}")

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.batch_list.count()):
            self.batch_list.item(row).setCheckState(state)

    def _checked_cases(self) -> list[str]:
        return [
            self.batch_list.item(row).text()
            for row in range(self.batch_list.count())
            if self.batch_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def _set_busy(self, busy: bool) -> None:
        self.inference_busy = busy
        self.refresh_button.setEnabled(not busy)
        self.batch_cancel_button.setEnabled(busy)
        self._update_run_buttons()

    def _update_run_buttons(self) -> None:
        if self.inference_busy:
            reason = "Inference is already running."
        elif not self.server_up:
            reason = server_problem_hint(self.server_message)
        else:
            reason = ""
        for button in (self.run_button, self.batch_run_button):
            button.setEnabled(not reason)
            button.setToolTip(reason)
        self.batch_resume_button.setEnabled(not reason and bool(self.remaining_cases))

    def start_single(self) -> None:
        case_id = self.case_box.currentData()
        if not case_id:
            self.append("No case selected.")
            return
        self._start_worker(batch.BatchPlan([case_id]))

    def start_batch(self) -> None:
        cases = self._checked_cases()
        if not cases:
            self.append("No cases checked.")
            return
        self.append(f"Batch started: {len(cases)} case(s) queued.")
        self._start_worker(self._plan(cases))

    def resume_batch(self) -> None:
        if not self.remaining_cases:
            self.append("Nothing to resume.")
            return
        self.append(f"Resuming batch: {len(self.remaining_cases)} case(s) left.")
        self._start_worker(self._plan(self.remaining_cases))

    def _plan(self, cases: list[str]) -> batch.BatchPlan:
        done = batch.completed_cases(
            history.scan(self.settings.results_path()), self.settings.model, self.settings.server_url
        )
        planned = batch.plan(cases, done, self.skip_existing_box.isChecked())
        for case_id in planned.skipped:
            self.append(f"Skipped {case_id} — result already exists.")
        return planned

    def _start_worker(self, planned: batch.BatchPlan) -> None:
        self.remaining_cases = []
        self._set_busy(True)
        self.batch_progress.setMaximum(max(len(planned.to_run), 1))
        self.batch_progress.setValue(0)
        self.count_label.setText(f"0 / {len(planned.to_run)}")
        self.batch_started_at = time.monotonic()
        self.case_started_at = None
        self.timer.start()

        self.thread = QThread()
        self.worker = batch.BatchWorker(self.backend, planned)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.case_done.connect(self.on_case_done)
        self.worker.case_failed.connect(self.on_case_failed)
        self.worker.finished.connect(self.on_all_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def cancel_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.append("Cancelling after the current case finishes...")

    def tick(self) -> None:
        if self.case_started_at is not None:
            self.case_timer_label.setText(format_duration(time.monotonic() - self.case_started_at))
        if self.batch_started_at is not None:
            self.total_timer_label.setText(
                f"total {format_duration(time.monotonic() - self.batch_started_at)}"
            )

    def on_progress(self, index: int, total: int, case_id: str) -> None:
        self.batch_progress.setValue(index)
        self.count_label.setText(f"{index} / {total}")
        self.case_started_at = time.monotonic()
        self.case_timer_label.setText("00:00")
        self.activity_label.setText(f"Running inference on {case_id}")
        self.activity_label.setStyleSheet(f"color: {theme.AMBER};")
        prefix = "" if total == 1 else f"[{index}/{total}] "
        self.append(f"{prefix}Running inference on {case_id}...")

    def on_case_done(self, case_id: str, elapsed: float) -> None:
        self.append(f"Done: {case_id} in {format_duration(elapsed)}")

    def on_case_failed(self, case_id: str, message: str) -> None:
        self.append(f"FAILED: {case_id} — {message}")

    def on_all_finished(self, summary: batch.BatchSummary) -> None:
        self.timer.stop()
        self.case_started_at = None
        total_time = format_duration(time.monotonic() - self.batch_started_at)
        self.batch_started_at = None
        self.case_timer_label.setText("")
        self.total_timer_label.setText("")
        self.activity_label.setText("Idle")
        self.activity_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.append(
            f"{'Cancelled' if summary.cancelled else 'Finished'} in {total_time} — "
            f"{summary.completed} completed, {summary.skipped} skipped, {summary.failed} failed."
        )
        self.remaining_cases = summary.remaining
        if summary.remaining:
            self.append(f"{len(summary.remaining)} case(s) not run. Press Resume to continue from "
                        f"{summary.remaining[0]}.")
        self._set_busy(False)
        self.reload_results()

    def import_dicom(self) -> None:
        studies = self.settings.studies_path()
        if studies is None:
            self.append("Set the studies folder in Settings first; imported scans are saved there.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM series folder", str(studies))
        if not folder:
            return
        self.import_button.setEnabled(False)
        self.append(f"Converting {folder} to NRRD...")
        self.importer.start(Path(folder), studies)

    def on_dicom_imported(self, path: Path) -> None:
        self.import_button.setEnabled(True)
        self.append(f"Saved {path}. Press Refresh once the server has picked it up to see it in the case list.")

    def on_dicom_failed(self, message: str) -> None:
        self.import_button.setEnabled(True)
        self.append(message)

    def open_slicer(self) -> None:
        case_id = self.case_box.currentData()
        if not case_id:
            self.append("No case selected.")
            return
        try:
            self.backend.open_in_slicer(case_id)
            self.append(f"Launching Slicer for {case_id}...")
        except Exception as exc:
            self.append(f"Could not launch Slicer: {exc}")

    def closeEvent(self, event) -> None:
        if self.inference_busy or self.compare_tab.busy:
            job = "An inference job" if self.inference_busy else "A comparison"
            answer = QMessageBox.question(
                self, "Job running",
                f"{job} is still running. Close anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer == QMessageBox.No:
                event.ignore()
                return
        if self.server.running:
            self.server.detach()
        if self.status_thread is not None:
            self.status_thread.quit()
            self.status_thread.wait(5000)
        self.setup_tab.close_thread()
        self.importer.close()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    logs.configure()
    window = MainWindow(Settings.load(), first_run=not CONFIG_FILE.exists())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
