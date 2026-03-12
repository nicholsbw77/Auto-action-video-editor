"""Resolve the application root directory for config, logs, and temp data.

When running as a PyInstaller bundle, sys.executable points to the .exe file,
so its parent is the directory containing the executable. In development,
we use the src/ directory (where this file lives).
"""
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Return the directory where config.json, logs/, and temp/ should live.

    - Frozen (PyInstaller --onedir): directory containing the .exe
    - Development: the src/ directory (parent of this module)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent
