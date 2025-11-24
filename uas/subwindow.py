"""
Base subwindow class for UAS framework.

Developers inherit from UASSubWindow to create custom visualization
and tool windows.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self
import uuid

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, QRect

if TYPE_CHECKING:
    from .main_window import UASMainWindow


class UASSubWindow(QWidget):
    """
    Base class for all subwindows in the UAS framework.
    
    Purpose:
        Provides the foundation for all subwindows with lifecycle management,
        serialization support, status messaging, and integration with main window
        display modes (MDI/Tabbed).
    
    Flow:
        1. Initializes with reference to parent main window and unique ID
        2. Calls on_create() after initialization for subclasses to set up UI
        3. on_activate()/on_deactivate() called when subwindow becomes active/inactive
        4. Supports serialization/deserialization for session persistence
        5. Can send status messages to main window's status bar
        6. On close, calls on_close() for cleanup and removes itself from main window
    """

    status_message = Signal(str)

    def __init__(self, main_window: UASMainWindow, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._id = str(uuid.uuid4())
        self._title = "Untitled"
        self._mdi_geometry: dict[str, int] = {}
        self.on_create()

    @property
    def main_window(self) -> UASMainWindow:
        """Get the parent main window that contains this subwindow."""
        return self._main_window

    @property
    def subwindow_id(self) -> str:
        """Get the unique identifier for this subwindow."""
        return self._id

    @property
    def title(self) -> str:
        """Get the subwindow title."""
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        """Set the subwindow title."""
        self._title = value
        self.setWindowTitle(value)


    @property
    def mdi_geometry(self) -> dict[str, int]:
        """Get the stored MDI geometry."""
        return self._mdi_geometry


    @mdi_geometry.setter
    def mdi_geometry(self, value: dict[str, int] | QRect | None) -> None:
        """Set the stored MDI geometry."""
        if value is None:
            self._mdi_geometry = {}
        elif isinstance(value, QRect):
            self._mdi_geometry = {
                "x": value.x(),
                "y": value.y(),
                "width": value.width(),
                "height": value.height(),
            }
        else:
            self._mdi_geometry = value

    type_name: str  # Class attribute: unique identifier for this subwindow type

    @classmethod
    def create(cls, main_window: UASMainWindow, state: dict[str, Any] | None = None) -> Self:
        """Factory method to create a new subwindow instance, optionally restoring from state.

        This allows the class to act as its own factory. Subclasses can override
        this method if they need custom creation logic.
        """
        subwindow = cls(main_window)
        if state:
            subwindow.deserialize(state)
        return subwindow

    def on_create(self) -> None:
        """Called when the subwindow is first created (override to set up UI)."""
        pass

    def on_activate(self) -> None:
        """Called when this subwindow becomes the active window (override to handle activation)."""
        pass

    def on_deactivate(self) -> None:
        """Called when this subwindow loses focus (override to handle deactivation)."""
        pass

    def on_close(self) -> None:
        """Called when the subwindow is about to close (override to release resources)."""
        pass

    def serialize(self) -> dict[str, Any]:
        """Serialize this subwindow's state to a dict (override to add custom state)."""
        state = {
            "type": self.type_name,
            "id": self._id,
            "title": self._title,
        }
        if self._mdi_geometry:
            state["mdi_geometry"] = self._mdi_geometry
        return state

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore this subwindow's state from a dict (override to restore custom state)."""
        if "id" in state:
            self._id = state["id"]
        if "title" in state:
            self.title = state["title"]
        if "mdi_geometry" in state:
            self._mdi_geometry = state["mdi_geometry"]

    def update_status(self, message: str) -> None:
        """Send a status message to the main window's status bar."""
        self.status_message.emit(message)

    def clone(self) -> UASSubWindow:
        """Create a clone of this subwindow by serializing and creating new instance."""
        from .factory import FactoryRegistry

        state = self.serialize()
        state["id"] = str(uuid.uuid4())
        cls = FactoryRegistry.get_instance().get_subwindow_class(self.type_name)
        return cls.create(self._main_window, state)

    def closeEvent(self, event) -> None:
        """Handle the close event by calling on_close() and removing from main window."""
        self.on_close()
        self._main_window.remove_subwindow(self)
        super().closeEvent(event)
