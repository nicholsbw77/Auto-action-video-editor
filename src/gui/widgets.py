import re
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import pyqtSignal

TIMECODE_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{2})")


def parse_timecode(tc: str) -> float:
    m = TIMECODE_RE.match(tc)
    if not m:
        raise ValueError(f"Invalid timecode: {tc}")
    h, mn, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mn * 60 + s + cs / 100.0


def format_timecode(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    seconds %= 60
    s = int(seconds)
    cs = int(round((seconds - s) * 100))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


class TimecodeInput(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if label:
            layout.addWidget(QLabel(label))
        self._edit = QLineEdit("00:00:00.00")
        self._edit.setInputMask("99:99:99.99")
        self._edit.textChanged.connect(self._on_changed)
        layout.addWidget(self._edit)

    def _on_changed(self, text: str):
        try:
            val = parse_timecode(text)
            self.value_changed.emit(val)
        except ValueError:
            pass

    def get_value(self) -> float:
        try:
            return parse_timecode(self._edit.text())
        except ValueError:
            return 0.0

    def set_value(self, seconds: float):
        self._edit.setText(format_timecode(seconds))


class FileListWidget(QListWidget):
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._paths: list[str] = []

    def add_file(self, path: str) -> bool:
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.VIDEO_EXTENSIONS:
            return False
        if path in self._paths:
            return False
        self._paths.append(path)
        self.addItem(QListWidgetItem(os.path.basename(path)))
        return True

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def clear_all(self):
        self._paths.clear()
        self.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        import os
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and not os.path.isdir(path):
                self.add_file(path)
