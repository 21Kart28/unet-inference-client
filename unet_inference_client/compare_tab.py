"""Compare tab: Dice, volume and HD95 between predictions and ground truth."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unet_inference_client import metrics, theme
from unet_inference_client.config import Settings
from unet_inference_client.history import StoredResult

IMAGE_FILTER = "Label maps (*.nrrd *.nii *.nii.gz);;All files (*)"
CSV_FIELDS = [
    "timestamp", "case_id", "prediction_path", "ground_truth_path", "label", "dice",
    "pred_volume_mm3", "gt_volume_mm3", "volume_diff_pct", "hd95_mm", "model", "checkpoint_epoch",
]


def format_dice(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def format_volume(value: float | None) -> str:
    return "" if value is None else f"{value:,.1f}"


def format_percent(value: float | None) -> str:
    return "" if value is None else f"{value:+.2f}"


def format_mm(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


VALUE_ROWS = [
    ("Dice", "dice", format_dice),
    ("Prediction volume (mm³)", "pred_volume_mm3", format_volume),
    ("Ground truth volume (mm³)", "gt_volume_mm3", format_volume),
    ("Volume difference (%)", "volume_diff_pct", format_percent),
    ("HD95 (mm)", "hd95_mm", format_mm),
]
TABLE_COLUMNS = [
    ("Case", "case_id", str),
    ("Dice", "dice", format_dice),
    ("Pred vol (mm³)", "pred_volume_mm3", format_volume),
    ("GT vol (mm³)", "gt_volume_mm3", format_volume),
    ("Vol diff (%)", "volume_diff_pct", format_percent),
    ("HD95 (mm)", "hd95_mm", format_mm),
]


def describe_error(exc: Exception) -> str:
    if isinstance(exc, metrics.GeometryMismatch):
        return (f"Can't compare. {exc} Check that the ground truth is for this case,"
                " or resample one file to match the other.")
    if isinstance(exc, FileNotFoundError):
        return f"{exc}. Check the path, or pick the file again with Browse."
    if isinstance(exc, ValueError):
        return f"{exc} Use a 3D label map saved as .nrrd, .nii or .nii.gz."
    if isinstance(exc, MemoryError):
        return "Not enough memory to load these volumes. Close other programs and try again."
    # SimpleITK errors carry a multi-line trace; the last line is the readable part.
    lines = str(exc).strip().splitlines()
    return lines[-1] if lines else type(exc).__name__


def metrics_rows(results: list[StoredResult]) -> list[dict]:
    rows = []
    for result in results:
        saved = metrics.load_metrics(result.output_dir)
        if saved is None:
            continue
        rows.append({
            "timestamp": saved.get("computed_at", ""),
            "case_id": result.case_id,
            "prediction_path": str(result.prediction_path),
            "ground_truth_path": saved.get("ground_truth_path", ""),
            "label": saved.get("label"),
            "dice": saved.get("dice"),
            "pred_volume_mm3": saved.get("pred_volume_mm3"),
            "gt_volume_mm3": saved.get("gt_volume_mm3"),
            "volume_diff_pct": saved.get("volume_diff_pct"),
            "hd95_mm": saved.get("hd95_mm"),
            "model": result.model,
            "checkpoint_epoch": result.epoch,
        })
    return rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Job:
    case_id: str
    prediction: Path
    ground_truth: Path | None
    save_to: Path | None


class CompareWorker(QObject):
    job_started = Signal(int, int, str)
    job_done = Signal(object, object)
    job_failed = Signal(object, str)
    finished = Signal()

    def __init__(self, jobs: list[Job], label: int | None, include_hd95: bool):
        super().__init__()
        self.jobs = jobs
        self.label = label
        self.include_hd95 = include_hd95
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def run(self) -> None:
        for index, job in enumerate(self.jobs, start=1):
            if self.cancelled:
                break
            self.job_started.emit(index, len(self.jobs), job.case_id)
            try:
                result = self._measure(job)
            except Exception as exc:
                self.job_failed.emit(job, describe_error(exc))
            else:
                self.job_done.emit(job, result)
        self.finished.emit()

    def _measure(self, job: Job) -> metrics.MaskMetrics | float:
        if job.ground_truth is None:
            return metrics.volume_mm3(job.prediction, self.label)
        result = metrics.compare(job.prediction, job.ground_truth, self.label, self.include_hd95)
        if job.save_to is not None:
            metrics.save_metrics(job.save_to, result, job.ground_truth, self.label)
        return result


class CompareTab(QWidget):
    metrics_saved = Signal()

    def __init__(self, settings: Settings, log: Callable[[str], None]):
        super().__init__()
        self.settings = settings
        self.log = log
        self.results = []
        self.worker_thread = None
        self.worker = None
        self.busy = False
        self.batch_mode = False
        self.started_at = 0.0
        self.compared = 0
        self.failures = []
        self.skipped = []
        self.saved_any = False

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_single())
        layout.addWidget(self._dim_label("All results"))
        layout.addWidget(self._build_table(), stretch=1)
        layout.addLayout(self._build_batch_buttons())
        layout.addWidget(self.summary_label)

    def _build_single(self) -> QHBoxLayout:
        self.prediction_edit = QLineEdit()
        self.prediction_edit.setPlaceholderText("pred.nrrd from a result folder, or any label map")
        self.ground_truth_edit = QLineEdit()
        self.ground_truth_edit.setPlaceholderText("Leave blank to measure the prediction volume only")
        self.ground_truth_hint = QLabel("")
        self.ground_truth_hint.setStyleSheet(f"color: {theme.AMBER};")
        self.ground_truth_hint.setWordWrap(True)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("any nonzero")
        self.label_edit.setFixedWidth(110)
        self.hd95_box = QCheckBox("Include HD95 (slower)")
        self.hd95_box.setChecked(True)

        options = QHBoxLayout()
        options.addWidget(self.label_edit)
        options.addSpacing(16)
        options.addWidget(self.hd95_box)
        options.addStretch()

        self.form = QFormLayout()
        self.form.setSpacing(8)
        self.form.addRow("Prediction", self._file_row(self.prediction_edit, self._browse_prediction))
        self.form.addRow("Ground truth", self._file_row(self.ground_truth_edit, self._browse_ground_truth))
        self.form.addRow("", self.ground_truth_hint)
        self.form.addRow("Label value", options)
        self._set_hint("")

        self.compute_button = QPushButton("Compute")
        self.compute_button.clicked.connect(self.compute)
        self.busy_bar = QProgressBar()
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-family: monospace;")

        actions = QHBoxLayout()
        actions.addWidget(self.compute_button)
        actions.addWidget(self.busy_bar, stretch=1)
        actions.addStretch()
        actions.addWidget(self.elapsed_label)

        left = QVBoxLayout()
        left.addLayout(self.form)
        left.addLayout(actions)
        left.addStretch()

        values = QFormLayout()
        values.setVerticalSpacing(6)
        values.setHorizontalSpacing(18)
        self.value_labels = []
        for title, key, formatter in VALUE_ROWS:
            value = QLabel("—")
            value.setStyleSheet("font-family: monospace;")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            values.addRow(self._dim_label(title), value)
            self.value_labels.append((key, formatter, value))
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)

        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(12, 10, 12, 10)
        box_layout.addLayout(values)
        box_layout.addWidget(self.message_label)
        box_layout.addStretch()
        box = QFrame()
        box.setObjectName("metricsBox")
        box.setLayout(box_layout)
        box.setStyleSheet(
            f"#metricsBox {{ background-color: {theme.SURFACE};"
            f" border: 1px solid {theme.BORDER}; border-radius: 6px; }}"
            " #metricsBox QLabel { background-color: transparent; }"
        )

        top = QHBoxLayout()
        top.addLayout(left, stretch=3)
        top.addWidget(box, stretch=2)
        return top

    def _build_table(self) -> QTableWidget:
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([title for title, _, _ in TABLE_COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMinimumHeight(130)
        return self.table

    def _build_batch_buttons(self) -> QHBoxLayout:
        self.compare_all_button = QPushButton("Compare all results")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QPushButton("Export metrics CSV")
        self.compare_all_button.clicked.connect(self.compare_all)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button.clicked.connect(self.export_csv)
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.summary_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.compare_all_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()
        buttons.addWidget(self.export_button)
        return buttons

    def _file_row(self, line_edit: QLineEdit, browse: Callable[[], None]) -> QWidget:
        button = QPushButton("Browse")
        button.clicked.connect(browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(button)
        container = QWidget()
        container.setLayout(row)
        return container

    def _dim_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        return label

    def set_results(self, results: list[StoredResult]) -> None:
        self.results = results
        self._fill_table()

    def load_result(self, result: StoredResult) -> None:
        self.prediction_edit.setText(str(result.prediction_path))
        self._fill_ground_truth(result.case_id)
        self._show(None)

    def compute(self) -> None:
        try:
            label = self._label()
        except ValueError as exc:
            self._set_message(str(exc), error=True)
            return
        prediction = _path_from(self.prediction_edit)
        if prediction is None:
            self._set_message("Choose a prediction file first.", error=True)
            return
        ground_truth = _path_from(self.ground_truth_edit)
        result = self._result_for(prediction)
        case_id = result.case_id if result else prediction.name
        save_to = result.output_dir if result and ground_truth else None
        self._show(None)
        self._start([Job(case_id, prediction, ground_truth, save_to)], label, batch=False)

    def compare_all(self) -> None:
        try:
            label = self._label()
        except ValueError as exc:
            self.summary_label.setText(str(exc))
            return
        studies = self.settings.studies_path()
        if studies is None:
            self.summary_label.setText("Set the Studies folder in Settings first.")
            return
        jobs, self.skipped = [], []
        for result in self.results:
            ground_truth = metrics.find_ground_truth(studies, result.case_id)
            if ground_truth is None:
                self.skipped.append(result.case_id)
            else:
                jobs.append(Job(result.case_id, result.prediction_path, ground_truth, result.output_dir))
        if not jobs:
            self.summary_label.setText(
                f"No results have ground truth in {studies / 'labels' / 'final'}."
                " Name each label file after its case, like CASE_001.nrrd."
            )
            return
        self.log(f"Comparing {len(jobs)} result(s) against ground truth...")
        self._start(jobs, label, batch=True)

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.log("Cancelling after the current case finishes...")

    def export_csv(self) -> None:
        rows = metrics_rows(self.results)
        if not rows:
            self.summary_label.setText("No saved metrics yet. Run Compute or Compare all results first.")
            return
        default = str(self.settings.results_path() / "metrics.csv")
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export metrics", default, "CSV files (*.csv)"
        )
        if not destination:
            return
        try:
            write_metrics_csv(Path(destination), rows)
        except OSError as exc:
            self.log(f"Could not write {destination}: {exc.strerror or exc}."
                     " If the file is open in Excel, close it and try again.")
            return
        self.log(f"Metrics for {len(rows)} result(s) exported to {destination}")

    def _start(self, jobs: list[Job], label: int | None, batch: bool) -> None:
        # finished fires just before the old thread exits, and dropping a running QThread aborts.
        if self.worker_thread is not None:
            self.worker_thread.wait()
        self.batch_mode = batch
        self.compared = 0
        self.failures = []
        self.saved_any = False
        if batch:
            self.busy_bar.setRange(0, len(jobs))
            self.busy_bar.setValue(0)
        else:
            self.busy_bar.setRange(0, 0)

        self.worker_thread = QThread()
        self.worker = CompareWorker(jobs, label, self.hd95_box.isChecked())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.job_started.connect(self._on_job_started)
        self.worker.job_done.connect(self._on_job_done)
        self.worker.job_failed.connect(self._on_job_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._on_finished)
        self._set_busy(True)
        self.worker_thread.start()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.compute_button.setEnabled(not busy)
        self.compare_all_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy and self.batch_mode)
        self.busy_bar.setVisible(busy)
        if busy:
            self.started_at = time.monotonic()
            self.timer.start()
        else:
            self.timer.stop()
        self._tick()

    def _tick(self) -> None:
        elapsed = time.monotonic() - self.started_at
        if elapsed < 60:
            self.elapsed_label.setText(f"{elapsed:.1f} s")
        else:
            minutes, seconds = divmod(int(elapsed), 60)
            self.elapsed_label.setText(f"{minutes}:{seconds:02d}")

    def _on_job_started(self, index: int, total: int, case_id: str) -> None:
        if self.batch_mode:
            self.busy_bar.setValue(index - 1)
            self.summary_label.setText(f"Comparing {case_id} ({index}/{total})")

    def _on_job_done(self, job: Job, result: metrics.MaskMetrics | float) -> None:
        self.compared += 1
        if isinstance(result, metrics.MaskMetrics):
            line = f"{job.case_id}: Dice {format_dice(result.dice)}"
            if result.hd95_mm is not None:
                line += f", HD95 {format_mm(result.hd95_mm)} mm"
            if result.note:
                line += f". {result.note}"
            self.log(line)
        else:
            self.log(f"{job.case_id}: prediction volume {format_volume(result)} mm³")
        if job.save_to is not None:
            self.saved_any = True
            self._fill_table()
        if not self.batch_mode:
            self._show(result)

    def _on_job_failed(self, job: Job, message: str) -> None:
        self.failures.append(job.case_id)
        self.log(f"{job.case_id}: {message}")
        if not self.batch_mode:
            self._set_message(message, error=True)

    def _on_finished(self) -> None:
        cancelled = self.worker.cancelled
        self._set_busy(False)
        if self.batch_mode:
            lines = [f"{'Cancelled. ' if cancelled else ''}Compared {self.compared},"
                     f" failed {len(self.failures)}, skipped {len(self.skipped)}."]
            if self.skipped:
                lines.append("Skipped (no ground truth): " + ", ".join(self.skipped))
            if self.failures:
                lines.append("Failed (see log): " + ", ".join(self.failures))
            self.summary_label.setText("\n".join(lines))
            for line in lines:
                self.log(line)
        if self.saved_any:
            self.metrics_saved.emit()

    def _show(self, result: metrics.MaskMetrics | float | None) -> None:
        if isinstance(result, metrics.MaskMetrics):
            values = asdict(result)
            message = result.note
        elif result is not None:
            values = {"pred_volume_mm3": result}
            message = "No ground truth chosen, so only the prediction volume was measured."
        else:
            values, message = {}, ""
        for key, formatter, label in self.value_labels:
            label.setText(formatter(values.get(key)) or "—")
        self._set_message(message)

    def _set_message(self, text: str, error: bool = False) -> None:
        self.message_label.setText(text)
        color = theme.RED if error else theme.TEXT_DIM
        self.message_label.setStyleSheet(f"color: {color};")

    def _fill_table(self) -> None:
        rows = metrics_rows(self.results)
        self.table.setRowCount(len(rows))
        for row, data in enumerate(rows):
            for column, (_, key, formatter) in enumerate(TABLE_COLUMNS):
                self.table.setItem(row, column, QTableWidgetItem(formatter(data[key]) or "—"))

    def _fill_ground_truth(self, case_id: str) -> None:
        studies = self.settings.studies_path()
        found = metrics.find_ground_truth(studies, case_id) if studies else None
        self.ground_truth_edit.setText(str(found) if found else "")
        if studies is None:
            self._set_hint("Set the Studies folder in Settings to find ground truth.")
        elif found is None:
            self._set_hint(f"No ground truth found for {case_id}. Pick one with Browse,"
                           " or leave it blank to measure volume only.")
        else:
            self._set_hint("")

    def _set_hint(self, text: str) -> None:
        self.ground_truth_hint.setText(text)
        self.form.setRowVisible(self.ground_truth_hint, bool(text))

    def _result_for(self, prediction: Path) -> StoredResult | None:
        target = prediction.resolve()
        return next((r for r in self.results if r.prediction_path.resolve() == target), None)

    def _label(self) -> int | None:
        text = self.label_edit.text().strip()
        if not text:
            return None
        if not text.isdigit() or int(text) == 0:
            raise ValueError("Label value must be a whole number above 0, or blank for any nonzero voxel.")
        return int(text)

    def _browse_prediction(self) -> None:
        start = self.prediction_edit.text() or str(self.settings.results_path())
        path, _ = QFileDialog.getOpenFileName(self, "Select prediction", start, IMAGE_FILTER)
        if not path:
            return
        self.prediction_edit.setText(path)
        result = self._result_for(Path(path))
        if result is not None:
            self._fill_ground_truth(result.case_id)
        else:
            self._set_hint("")

    def _browse_ground_truth(self) -> None:
        studies = self.settings.studies_path()
        start = self.ground_truth_edit.text() or (str(studies / "labels" / "final") if studies else "")
        path, _ = QFileDialog.getOpenFileName(self, "Select ground truth", start, IMAGE_FILTER)
        if path:
            self.ground_truth_edit.setText(path)
            self._set_hint("")


def _path_from(line_edit: QLineEdit) -> Path | None:
    # Explorer's "Copy as path" wraps the path in quotes.
    text = line_edit.text().strip().strip('"')
    return Path(text) if text else None
