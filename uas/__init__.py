"""
Universal Application Shell (UAS) Framework

A PySide6-based framework for building interactive visualization applications
with flexible window management and session persistence.
"""

from .factory import (
    FactoryRegistry,
    auto_register,
)
from .subwindow import UASSubWindow
from .main_window import UASMainWindow, DisplayMode
from .session import SessionManager, UASApplication
from .visualization import UASVisualizationSubWindow, UASGraphicsView
from .threading import (
    ThreadStatus,
    BackgroundWorker,
    ThreadedSubWindowMixin,
)
from .observer import (
    DataEvent,
    ObservableSubWindowMixin,
    ListenerSubWindowMixin,
)
from .shortcuts import (
    ShortcutManager,
    ShortcutMixin,
    install_shortcut_handler,
)

__all__ = [
    # Factory system
    "FactoryRegistry",
    "auto_register",
    # Base classes
    "UASSubWindow",
    "UASMainWindow",
    "DisplayMode",
    # Session management
    "SessionManager",
    "UASApplication",
    # Visualization
    "UASVisualizationSubWindow",
    "UASGraphicsView",
    # Threading
    "ThreadStatus",
    "BackgroundWorker",
    "ThreadedSubWindowMixin",
    # Observer pattern
    "DataEvent",
    "ObservableSubWindowMixin",
    "ListenerSubWindowMixin",
    # Shortcuts
    "ShortcutManager",
    "ShortcutMixin",
    "install_shortcut_handler",
]

__version__ = "0.1.0"
