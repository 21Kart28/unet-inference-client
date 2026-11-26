"""Models tab: Dice per case for two models side by side, with the difference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from unet_inference_client import model_compare, report, theme
from unet_inference_client.compare_tab import format_dice
from unet_inference_client.config import Settings
from unet_inference_client.history import StoredResult

HEADERS = ["Case", "Dice A", "Dice B", "Delta (B - A)"]


def format_delta(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


class ModelCompareTab(QWidget):
    def __init__(self, settings: Settings, log: Callable[[str], None]):
        super().__init__()
        self.settings = settings
        self.log = log
        self.results: list[StoredResult] = []
        self.rows: list[model_compare.DeltaRow] = []

        self.model_a_box = QComboBox()
        self.model_b_box = QComboBox()
        for box in (self.model_a_box, self.model_b_box):
            box.setMinimumWidth(180)
            box.currentIndexChanged.connect(self.refresh)
        self.export_button = QPushButton("Export deltas CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.report_button = QPushButton("Export report")
        self.report_button.clicked.connect(self.export_report)

        pickers = QHBoxLayout()
        pickers.addWidget(self._dim_label("Model A"))
        pickers.addWidget(self.model_a_box)
        pickers.addSpacing(16)
        pickers.addWidget(self._dim_label("Model B"))
        pickers.addWidget(self.model_b_box)
        pickers.addStretch()
        pickers.addWidget(self.export_button)
        pickers.addWidget(self.report_button)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {theme.TEXT_DIM};")

        layout = QVBoxLayout(self)
        layout.addLayout(pickers)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.summary_label)

    def set_results(self, results: list[StoredResult]) -> None:
        self.results = results
        models = sorted(model_compare.dice_by_model(results))
        for box in (self.model_a_box, self.model_b_box):
            current = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems(models)
            if current in models:
                box.setCurrentText(current)
            box.blockSignals(False)
        if len(models) > 1 and self.model_a_box.currentText() == self.model_b_box.currentText():
            self.model_b_box.blockSignals(True)
            self.model_b_box.setCurrentIndex(1)
            self.model_b_box.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        model_a, model_b = self.model_a_box.currentText(), self.model_b_box.currentText()
        if not model_a or not model_b:
            self.rows = []
            self.table.setRowCount(0)
            self.summary_label.setText(
                "No saved Dice yet. Run a batch with each model, then Compare all results."
            )
            return
        self.rows = model_compare.delta_rows(self.results, model_a, model_b)
        self.table.setRowCount(len(self.rows))
        for row, data in enumerate(self.rows):
            cells = [data.case_id, format_dice(data.dice_a), format_dice(data.dice_b),
                     format_delta(data.delta)]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text or "—")
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)
            if data.delta:
                self.table.item(row, 3).setForeground(QColor(theme.GREEN if data.delta > 0 else theme.RED))
        summary = model_compare.summarize(self.rows)
        self.summary_label.setText(model_compare.describe(summary, model_a, model_b))

    def export_csv(self) -> None:
        model_a, model_b = self.model_a_box.currentText(), self.model_b_box.currentText()
        if not self.rows:
            self.summary_label.setText("Nothing to export yet. Pick two models with saved Dice.")
            return
        default = str(self.settings.results_path() / f"dice_{model_a}_vs_{model_b}.csv")
        destination, _ = QFileDialog.getSaveFileName(self, "Export deltas", default, "CSV files (*.csv)")
        if not destination:
            return
        try:
            model_compare.write_delta_csv(Path(destination), self.rows, model_a, model_b)
        except OSError as exc:
            self.log(f"Could not write {destination}: {exc.strerror or exc}."
                     " If the file is open in Excel, close it and try again.")
            return
        self.log(f"Dice deltas for {len(self.rows)} case(s) exported to {destination}")

    def export_report(self) -> None:
        model_a, model_b = self.model_a_box.currentText(), self.model_b_box.currentText()
        if not self.rows:
            self.summary_label.setText("Nothing to export yet. Pick two models with saved Dice.")
            return
        default = str(self.settings.results_path() / f"models_{model_a}_vs_{model_b}.md")
        destination, _ = QFileDialog.getSaveFileName(self, "Export report", default, report.FILE_FILTER)
        if not destination:
            return
        document = report.from_deltas(
            self.rows, model_compare.summarize(self.rows), model_a, model_b, self.settings.server_url
        )
        try:
            report.write(Path(destination), document)
        except OSError as exc:
            self.log(f"Could not write {destination}: {exc.strerror or exc}.")
            return
        self.log(f"Comparison report for {len(self.rows)} case(s) exported to {destination}")

    def _dim_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        return label
