import sys
import os

# Ensure src/ is at position 0 in sys.path so project packages take priority
# over any test package directories that share the same name (e.g., tests/gui vs src/gui).
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
# Remove any existing entry and re-insert at position 0
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)
