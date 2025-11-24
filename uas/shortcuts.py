"""
Keyboard shortcut system with context-specific shortcuts and priority routing.

Active subwindow gets priority for shortcut handling, with fallback to
main window shortcuts.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
import warnings

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from .subwindow import UASSubWindow
    from .main_window import UASMainWindow


@dataclass
class ShortcutInfo:
    """
    Information about a registered shortcut.
    
    Purpose:
        Stores metadata for a keyboard shortcut including the key sequence,
        callback function, description, and context (global or subwindow type).
    """

    key_sequence: QKeySequence
    callback: Callable[[], None]
    description: str
    context: str


class ShortcutManager:
    """
    Manages keyboard shortcuts with context-specific routing.
    
    Purpose:
        Central manager for keyboard shortcuts with priority-based routing.
        Supports global shortcuts (always active) and subwindow-specific shortcuts
        (active only when that subwindow type is active). Active subwindow shortcuts
        take priority over global shortcuts.
    
    Flow:
        1. Shortcuts registered via register_global_shortcut() or register_subwindow_shortcut()
        2. Key events intercepted by install_shortcut_handler() on main window
        3. Manager looks up shortcut based on active subwindow type (priority: subwindow > global)
        4. If found, callback is executed; otherwise event propagates normally
    """

    _instance: ShortcutManager | None = None

    def __init__(self) -> None:
        """Initialize the shortcut manager with empty shortcut dictionaries."""
        self._global_shortcuts: dict[str, ShortcutInfo] = {}
        self._subwindow_shortcuts: dict[str, dict[str, ShortcutInfo]] = {}

    @classmethod
    def get_instance(cls) -> ShortcutManager:
        """Get the global shortcut manager instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = ShortcutManager()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the shortcut manager (mainly for testing)."""
        cls._instance = None

    def register_global_shortcut(
        self,
        key: str,
        callback: Callable[[], None],
        description: str = ""
    ) -> None:
        """Register a global shortcut that is always active."""
        key_seq = QKeySequence(key)
        key_str = key_seq.toString()

        if key_str in self._global_shortcuts:
            existing = self._global_shortcuts[key_str]
            warnings.warn(
                f"Shortcut '{key}' is already registered globally "
                f"for '{existing.description}'. Overwriting."
            )

        self._global_shortcuts[key_str] = ShortcutInfo(
            key_sequence=key_seq,
            callback=callback,
            description=description,
            context="global"
        )

    def register_subwindow_shortcut(
        self,
        subwindow_type: str,
        key: str,
        callback: Callable[[], None],
        description: str = ""
    ) -> None:
        """Register a context-specific shortcut for a subwindow type (active only when that type is active)."""
        key_seq = QKeySequence(key)
        key_str = key_seq.toString()

        if subwindow_type not in self._subwindow_shortcuts:
            self._subwindow_shortcuts[subwindow_type] = {}

        shortcuts = self._subwindow_shortcuts[subwindow_type]
        if key_str in shortcuts:
            existing = shortcuts[key_str]
            if existing.description == description:
                return
            warnings.warn(
                f"Shortcut '{key}' is already registered for subwindow type "
                f"'{subwindow_type}' for '{existing.description}'. Overwriting."
            )

        if key_str in self._global_shortcuts:
            warnings.warn(
                f"Shortcut '{key}' for subwindow '{subwindow_type}' will override "
                f"global shortcut '{self._global_shortcuts[key_str].description}' "
                f"when this subwindow type is active."
            )

        shortcuts[key_str] = ShortcutInfo(
            key_sequence=key_seq,
            callback=callback,
            description=description,
            context=subwindow_type
        )

    def unregister_global_shortcut(self, key: str) -> None:
        """Unregister a global shortcut."""
        key_str = QKeySequence(key).toString()
        self._global_shortcuts.pop(key_str, None)

    def unregister_subwindow_shortcut(self, subwindow_type: str, key: str) -> None:
        """Unregister a subwindow-specific shortcut."""
        key_str = QKeySequence(key).toString()
        if subwindow_type in self._subwindow_shortcuts:
            self._subwindow_shortcuts[subwindow_type].pop(key_str, None)

    def get_shortcut_for_context(
        self,
        key: str,
        active_subwindow_type: str | None
    ) -> ShortcutInfo | None:
        """Get the appropriate shortcut for the current context (subwindow shortcuts take priority)."""
        key_str = QKeySequence(key).toString()

        if active_subwindow_type and active_subwindow_type in self._subwindow_shortcuts:
            shortcuts = self._subwindow_shortcuts[active_subwindow_type]
            if key_str in shortcuts:
                return shortcuts[key_str]

        return self._global_shortcuts.get(key_str)

    def get_all_shortcuts(self) -> list[ShortcutInfo]:
        """Get all registered shortcuts (global and subwindow-specific)."""
        result = list(self._global_shortcuts.values())
        for shortcuts in self._subwindow_shortcuts.values():
            result.extend(shortcuts.values())
        return result

    def get_shortcuts_for_subwindow_type(self, subwindow_type: str) -> list[ShortcutInfo]:
        """Get all shortcuts for a specific subwindow type."""
        if subwindow_type not in self._subwindow_shortcuts:
            return []
        return list(self._subwindow_shortcuts[subwindow_type].values())


class ShortcutMixin:
    """
    Mixin that adds shortcut registration to a subwindow.
    
    Purpose:
        Provides convenient methods for subwindows to register context-specific
        keyboard shortcuts that are only active when the subwindow is active.
    
    Flow:
        1. Subwindow calls register_shortcut() in on_create() or elsewhere
        2. Mixin delegates to ShortcutManager with subwindow's type_name
        3. Shortcut is active only when this subwindow type is the active subwindow
    """

    def register_shortcut(
        self,
        key: str,
        callback: Callable[[], None],
        description: str = ""
    ) -> None:
        """Register a context-specific shortcut for this subwindow type (active only when this subwindow is active)."""
        manager = ShortcutManager.get_instance()
        manager.register_subwindow_shortcut(
            self.type_name,
            key,
            callback,
            description
        )

    def unregister_shortcut(self, key: str) -> None:
        """Unregister a shortcut for this subwindow type."""
        manager = ShortcutManager.get_instance()
        manager.unregister_subwindow_shortcut(self.type_name, key)


def install_shortcut_handler(main_window: UASMainWindow) -> None:
    """Install the shortcut handler on a main window to intercept key events and route through ShortcutManager."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    original_event = main_window.event

    def custom_event(event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key_event = event
            if isinstance(key_event, QKeyEvent):
                key_seq = QKeySequence(
                    int(key_event.modifiers()) | key_event.key()
                )
                key_str = key_seq.toString()

                active_type = None
                if main_window.active_subwindow:
                    active_type = main_window.active_subwindow.type_name

                manager = ShortcutManager.get_instance()
                shortcut = manager.get_shortcut_for_context(key_str, active_type)

                if shortcut:
                    shortcut.callback()
                    return True

        return original_event(event)

    main_window.event = custom_event
