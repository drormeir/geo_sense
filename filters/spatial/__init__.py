"""
Spatial domain filters.

Filters that operate across traces (horizontal direction) rather than
within individual traces. Auto-registered via @register_filter decorator.

Usage:
    python -m filters.spatial
"""

# Auto-import all filter modules (*.py except _*.py) to trigger registration
import importlib
from pathlib import Path

_package_dir = Path(__file__).parent
for _file in _package_dir.glob("*.py"):
    if not _file.name.startswith("_"):
        importlib.import_module(f".{_file.stem}", __package__)
