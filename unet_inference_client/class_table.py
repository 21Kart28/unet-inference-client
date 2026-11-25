"""Table of Dice and volume per label, shown under the Compare tab's single-case form."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from unet_inference_client.metrics import ClassMetrics

HEADERS = ["Label", "Dice", "Pred vol (mm³)", "GT vol (mm³)"]


class ClassTable(QTableWidget):
    def __init__(self):
        super().__init__(0, len(HEADERS))
        self.setHorizontalHeaderLabels(HEADERS)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setMaximumHeight(160)
        self.setVisible(False)

    def show_classes(self, classes: list[ClassMetrics]) -> None:
        self.setRowCount(len(classes))
        for row, item in enumerate(classes):
            cells = [str(item.label), f"{item.dice:.4f}",
                     f"{item.pred_volume_mm3:,.1f}", f"{item.gt_volume_mm3:,.1f}"]
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(row, column, cell)
        self.setVisible(bool(classes))

    def clear_classes(self) -> None:
        self.setRowCount(0)
        self.setVisible(False)
