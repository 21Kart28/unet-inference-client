"""Dark stylesheet for the application."""

BACKGROUND = "#0d1117"
SURFACE = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
CYAN = "#39d3d3"
AMBER = "#f2a33c"
GREEN = "#3fb950"
RED = "#f85149"
SELECTION = "#1b2b20"

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_DIM};
    padding: 7px 18px;
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {CYAN};
    border: 1px solid {BORDER};
    border-bottom-color: {SURFACE};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}
QPushButton {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    color: {TEXT};
}}
QPushButton:hover:enabled {{
    border-color: {GREEN};
    color: {GREEN};
}}
QPushButton:disabled {{
    color: #484f58;
    border-color: #21262d;
}}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px;
    selection-background-color: {SELECTION};
    selection-color: {GREEN};
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {GREEN};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {SELECTION};
    selection-color: {GREEN};
}}
QPlainTextEdit {{
    font-family: monospace;
    font-size: 12px;
}}
QListWidget::item {{
    padding: 3px;
}}
QListWidget::item:hover {{
    background-color: #1f242c;
}}
QProgressBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{
    background-color: {AMBER};
    border-radius: 4px;
}}
QHeaderView::section {{
    background-color: {SURFACE};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
}}
QTableWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: {BORDER};
}}
QTableWidget::item:selected {{
    background-color: {SELECTION};
    color: {GREEN};
}}
QCheckBox {{
    spacing: 7px;
}}
QCheckBox::indicator, QListWidget::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {TEXT_DIM};
    border-radius: 3px;
    background-color: {BACKGROUND};
}}
QCheckBox::indicator:hover, QListWidget::indicator:hover {{
    border-color: {GREEN};
}}
QCheckBox::indicator:checked, QListWidget::indicator:checked {{
    background-color: {GREEN};
    border-color: {GREEN};
}}
QScrollBar:vertical {{
    background: {BACKGROUND};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px;
}}
"""

