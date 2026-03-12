import subprocess
import sys


def _subprocess_flags() -> dict:
    """Return platform-specific subprocess kwargs to suppress console windows on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
