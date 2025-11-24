"""
Factory system for registering and creating window types.

Window classes act as their own factories by implementing a class-level
`type_name` attribute and a `create()` classmethod. These are registered
globally at application startup and identified by string names for serialization.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .subwindow import UASSubWindow
    from .main_window import UASMainWindow


class FactoryRegistry:
    """
    Global registry for all window factories.
    
    Purpose:
        Central registry that stores and retrieves factories for creating
        subwindows and main windows. Uses singleton pattern to provide
        global access throughout the application.
    
    Flow:
        1. Application startup: factories are registered via register_* methods
        2. During session load: registry looks up factories by type_name from serialized state
        3. Factory.create() is called to instantiate windows with optional state
        4. Supports querying registered types for UI menus or validation
    """

    _instance: FactoryRegistry | None = None

    def __init__(self) -> None:
        """Initialize the factory registry with empty dictionaries."""
        self._subwindow_factories: dict[str, type[UASSubWindow]] = {}
        self._main_window_factories: dict[str, type[UASMainWindow]] = {}

    @classmethod
    def get_instance(cls) -> FactoryRegistry:
        """Get the global factory registry instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = FactoryRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._instance = None

    def register_subwindow(self, cls: type[UASSubWindow]) -> None:
        """Register a subwindow class in the registry."""
        if cls.type_name in self._subwindow_factories:
            raise ValueError(f"Subwindow '{cls.type_name}' is already registered")
        self._subwindow_factories[cls.type_name] = cls

    def register_main_window(self, cls: type[UASMainWindow]) -> None:
        """Register a main window class in the registry."""
        if cls.type_name in self._main_window_factories:
            raise ValueError(f"Main window '{cls.type_name}' is already registered")
        self._main_window_factories[cls.type_name] = cls

    def get_subwindow_class(self, type_name: str) -> type[UASSubWindow]:
        """Get a subwindow class by type name."""
        if type_name not in self._subwindow_factories:
            raise KeyError(f"No subwindow registered for type '{type_name}'")
        return self._subwindow_factories[type_name]

    def get_main_window_class(self, type_name: str) -> type[UASMainWindow]:
        """Get a main window class by type name."""
        if type_name not in self._main_window_factories:
            raise KeyError(f"No main window registered for type '{type_name}'")
        return self._main_window_factories[type_name]

    def get_registered_subwindow_types(self) -> list[str]:
        """Get list of all registered subwindow type names."""
        return list(self._subwindow_factories.keys())

    def get_registered_main_window_types(self) -> list[str]:
        """Get list of all registered main window types."""
        return list(self._main_window_factories.keys())


def auto_register[T: (UASSubWindow, UASMainWindow)](cls: type[T]) -> type[T]:
    """
    Decorator that automatically registers a window class.

    Usage:
        @auto_register
        class MySubWindow(UASSubWindow):
            type_name = "my_subwindow"
            ...
    """
    from .subwindow import UASSubWindow
    from .main_window import UASMainWindow

    registry = FactoryRegistry.get_instance()
    if issubclass(cls, UASMainWindow):
        registry.register_main_window(cls)
    elif issubclass(cls, UASSubWindow):
        registry.register_subwindow(cls)
    return cls
