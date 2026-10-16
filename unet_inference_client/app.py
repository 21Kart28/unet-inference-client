"""Main window for U-Net Inference Client."""

import shutil
import sys
import time

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

from unet_inference_client import history, metrics, server, theme
from unet_inference_client.backend import Backend, MonaiLabelBackend
from unet_inference_client.compare_tab import CompareTab, format_dice, format_volume
from unet_inference_client.config import Settings


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def server_problem_hint(message: str) -> str:
    if message.startswith("unreachable"):
        return "Server unreachable. Check the Server URL in Settings, or start the server on the Server tab."
    return f"Server {message}. Check the inference model in Settings."


class BatchWorker(QObject):
    progress = Signal(int, int, str)
    case_done = Signal(str, float)
    case_failed = Signal(str, str)
    finished = Signal(int, int, int)

    def __init__(self, backend: Backend, case_ids: list[str], skip_existing: bool):
        super().__init__()
        self.backend = backend
        self.case_ids = case_ids
        self.skip_existing = skip_existing
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        completed = skipped = failed = 0
        total = len(self.case_ids)
        for index, case_id in enumerate(self.case_ids, start=1):
            if self.cancelled:
                break
            if self.skip_existing and self.backend.has_result(case_id):
                skipped += 1
                self.progress.emit(index, total, f"{case_id} (skipped)")
                continue
            self.progress.emit(index, total, case_id)
            try:
                result = self.backend.run_inference(case_id)
                completed += 1
                self.case_done.emit(case_id, result.elapsed_seconds)
            except Exception as exc:
                failed += 1
                self.case_failed.emit(case_id, str(exc))
        self.finished.emit(completed, skipped, failed)


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
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.backend = MonaiLabelBackend(settings)
        self.server = server.ServerProcess()
        self.thread = None
        self.worker = None
        self.reader_thread = None
        self.reader = None
        self.batch_started_at = None
        self.case_started_at = None
        self.stored_results = []
        self.inference_busy = False
        self.server_up = False
        self.server_message = None

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)

        self.setWindowTitle("U-Net Inference Client")
        self.resize(960, 740)

        self.status_label = QLabel("Server: checking...")
        self.refresh_button = QPushButton("Refresh")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        header = QHBoxLayout()
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(self.refresh_button)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_batch_tab(), "Batch")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.compare_tab = CompareTab(settings, self.append)
        self.compare_tab.metrics_saved.connect(self.reload_results)
        self.tabs.addTab(self.compare_tab, "Compare")
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
        strip.setLayout(layout)
        strip.setStyleSheet(
            f"QFrame {{ background-color: {theme.SURFACE};"
            f" border: 1px solid {theme.BORDER}; border-radius: 6px; }}"
        )
        return strip

    def _build_run_tab(self) -> QWidget:
        self.case_box = QComboBox()
        self.case_box.setMaxVisibleItems(12)
        self.run_button = QPushButton("Run inference")
        self.slicer_button = QPushButton("View in Slicer")
        self.run_button.clicked.connect(self.start_single)
        self.slicer_button.clicked.connect(self.open_slicer)

        row = QHBoxLayout()
        row.addWidget(QLabel("Case"))
        row.addWidget(self.case_box, stretch=1)
        row.addWidget(self.run_button)
        row.addWidget(self.slicer_button)

        hint = QLabel("Runs the model on one case. View opens an existing prediction in 3D Slicer.")
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
        self.batch_cancel_button = QPushButton("Cancel")
        self.batch_cancel_button.setEnabled(False)

        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        self.batch_run_button.clicked.connect(self.start_batch)
        self.batch_cancel_button.clicked.connect(self.cancel_batch)

        buttons = QHBoxLayout()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.select_none_button)
        buttons.addStretch()
        buttons.addWidget(self.batch_run_button)
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

    def save_server_settings(self) -> None:
        self.settings.server_python = self.server_python_edit.text().strip()
        self.settings.server_app_dir = self.server_app_edit.text().strip()
        self.settings.studies_dir = self.server_studies_edit.text().strip()
        self.settings.server_models = self.server_models_edit.text().strip()
        try:
            self.settings.server_port = int(self.server_port_edit.text().strip())
        except ValueError:
            raise RuntimeError("Port must be a number.")
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
        is_up, message = self.backend.server_status()
        self.status_label.setText(f"Server: {message}")
        color = theme.GREEN if is_up else theme.RED
        self.status_label.setStyleSheet(f"color: {color};")
        if message != self.server_message:
            self.append(f"Server: {message}")
        self.server_up, self.server_message = is_up, message
        self._update_run_buttons()
        if not is_up:
            self.case_box.clear()
            self.batch_list.clear()
            return
        try:
            cases = self.backend.list_cases()
        except Exception as exc:
            self.append(f"Could not load cases: {exc}")
            return
        self.case_box.clear()
        self.batch_list.clear()
        for case in cases:
            suffix = "" if case.has_ground_truth else "  (no ground truth)"
            self.case_box.addItem(f"{case.case_id}{suffix}", case.case_id)
            item = QListWidgetItem(case.case_id)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.batch_list.addItem(item)
        self.append(f"Loaded {len(cases)} cases.")

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

    def start_single(self) -> None:
        case_id = self.case_box.currentData()
        if not case_id:
            self.append("No case selected.")
            return
        self._start_worker([case_id], skip_existing=False)

    def start_batch(self) -> None:
        cases = self._checked_cases()
        if not cases:
            self.append("No cases checked.")
            return
        self.append(f"Batch started: {len(cases)} case(s) queued.")
        self._start_worker(cases, skip_existing=self.skip_existing_box.isChecked())

    def _start_worker(self, case_ids: list[str], skip_existing: bool) -> None:
        self._set_busy(True)
        self.batch_progress.setMaximum(len(case_ids))
        self.batch_progress.setValue(0)
        self.count_label.setText(f"0 / {len(case_ids)}")
        self.batch_started_at = time.monotonic()
        self.case_started_at = None
        self.timer.start()

        self.thread = QThread()
        self.worker = BatchWorker(self.backend, case_ids, skip_existing)
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
        if case_id.endswith("(skipped)"):
            self.append(f"Skipped {case_id.replace(' (skipped)', '')} — result already exists.")
            return
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

    def on_all_finished(self, completed: int, skipped: int, failed: int) -> None:
        self.timer.stop()
        self.case_started_at = None
        total_time = format_duration(time.monotonic() - self.batch_started_at)
        self.batch_started_at = None
        self.case_timer_label.setText("")
        self.total_timer_label.setText("")
        self.activity_label.setText("Idle")
        self.activity_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.append(
            f"Finished in {total_time} — {completed} completed, {skipped} skipped, {failed} failed."
        )
        self._set_busy(False)
        self.reload_results()

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
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow(Settings.load())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
