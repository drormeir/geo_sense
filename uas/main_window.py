"""
Base main window class for UAS framework.

Provides two display modes (MDI, Tabbed) and manages subwindows.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Self
import uuid

from PySide6.QtWidgets import (
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QTabWidget,
    QWidget,
    QStatusBar,
    QVBoxLayout,
    QToolBar,
    QDockWidget,
    QApplication,
)
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QAction, QCloseEvent

from .subwindow import UASSubWindow
from .factory import FactoryRegistry
from .session import SessionManager


class DisplayMode(Enum):
    """
    Display modes for main windows.
    
    Purpose:
        Enumeration of available display modes for organizing subwindows.
        MDI allows multiple overlapping windows, TABBED uses tab interface.
    """

    MDI = auto()
    TABBED = auto()


def fit_into_geometry(x: int, y: int, width: int, height: int, rect: QRect) -> tuple[int, int, int, int]:
    """Fit a rectangle into a geometry, ensuring it doesn't exceed the geometry's boundaries."""
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if rect.width() > 0:
        if width > rect.width():
            width = rect.width()
        if x + width > rect.width():
            x = rect.width() - width
    if rect.height() > 0:
        if height > rect.height():
            height = rect.height()
        if y + height > rect.height():
            y = rect.height() - height
    return x, y, width, height


class UASMainWindow(QMainWindow):
    """
    Base class for all main windows in the UAS framework.
    
    Purpose:
        Provides the foundation for main application windows with support for
        multiple display modes (MDI/Tabbed), subwindow management, toolbars,
        dock widgets, and session persistence.
    
    Flow:
        1. Initializes with MDI mode by default, sets up status bar and menus
        2. Manages list of subwindows and tracks active subwindow
        3. Supports switching between MDI and Tabbed display modes
        4. Handles subwindow lifecycle: add, remove, activate, deactivate
        5. Provides serialization/deserialization for session persistence
        6. Manages toolbars and dock widgets for extensible UI
        7. On close, saves session if auto-save enabled and it's the last window
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._id = str(uuid.uuid4())
        self._display_mode = DisplayMode.MDI
        self._subwindows: list[UASSubWindow] = []
        self._active_subwindow: UASSubWindow | None = None

        self._mdi_area: QMdiArea | None = None
        self._tab_widget: QTabWidget | None = None
        self._toolbars: dict[str, QToolBar] = {}
        self._dock_widgets: dict[str, QDockWidget] = {}
        self._central_container = QWidget()
        self._central_layout = QVBoxLayout(self._central_container)
        self._central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self._central_container)

        self._setup_display_mode(self._display_mode)
        self._setup_status_bar()
        self._setup_menus()
        self._setup_shortcuts()

        self.setWindowTitle("UAS Main Window")
        self.resize(1024, 768)


    @property
    def window_id(self) -> str:
        """Get the unique identifier for this main window."""
        return self._id

    type_name: str  # Class attribute: unique identifier for this main window type


    @classmethod
    def create(cls, state: dict[str, Any] | None = None) -> Self:
        """Factory method to create a new main window instance, optionally restoring from state.

        This allows the class to act as its own factory. Subclasses can override
        this method if they need custom creation logic.
        """
        window = cls()
        if state:
            window.deserialize(state)
        return window


    @property
    def display_mode(self) -> DisplayMode:
        """Get the current display mode."""
        return self._display_mode


    @property
    def subwindows(self) -> list[UASSubWindow]:
        """Get all subwindows in this main window."""
        return self._subwindows.copy()


    @property
    def active_subwindow(self) -> UASSubWindow | None:
        """Get the currently active subwindow."""
        return self._active_subwindow


    def set_display_mode(self, mode: DisplayMode) -> None:
        """Switch to a different display mode, preserving subwindow states."""
        if mode == self._display_mode:
            return

        self._update_mdi_geometries()
        subwindow_states = [sw.serialize() for sw in self._subwindows]
        active_id = self._active_subwindow.subwindow_id if self._active_subwindow else None

        for sw in self._subwindows[:]:
            self._remove_subwindow_from_display(sw)

        self._cleanup_display_mode()
        self._display_mode = mode
        self._setup_display_mode(mode)

        registry = FactoryRegistry.get_instance()
        for state in subwindow_states:
            cls = registry.get_subwindow_class(state["type"])
            subwindow = cls.create(self, state)
            self._add_subwindow_to_display(subwindow)
            self._subwindows.append(subwindow)
            if subwindow.subwindow_id == active_id:
                self._set_active_subwindow(subwindow)


    def _setup_display_mode(self, mode: DisplayMode) -> None:
        """Set up the UI components (MDI area or tab widget) for a display mode."""
        if mode == DisplayMode.MDI:
            self._mdi_area = QMdiArea()
            self._mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._mdi_area.subWindowActivated.connect(self._on_mdi_subwindow_activated)
            self._central_layout.addWidget(self._mdi_area)
        elif mode == DisplayMode.TABBED:
            self._tab_widget = QTabWidget()
            self._tab_widget.setTabsClosable(True)
            self._tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
            self._tab_widget.currentChanged.connect(self._on_tab_changed)
            self._central_layout.addWidget(self._tab_widget)


    def _cleanup_display_mode(self) -> None:
        """Clean up UI components from current display mode before switching."""
        if self._mdi_area:
            self._central_layout.removeWidget(self._mdi_area)
            self._mdi_area.deleteLater()
            self._mdi_area = None
        if self._tab_widget:
            self._central_layout.removeWidget(self._tab_widget)
            self._tab_widget.deleteLater()
            self._tab_widget = None
        self._subwindows.clear()


    def _setup_status_bar(self) -> None:
        """Set up the status bar for displaying messages from subwindows."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)


    def _setup_menus(self) -> None:
        """Set up the menu bar with standard menus (override to customize)."""
        menubar = self.menuBar()

        window_menu = menubar.addMenu("&Window")

        mdi_action = QAction("&MDI Mode", self)
        mdi_action.triggered.connect(lambda: self.set_display_mode(DisplayMode.MDI))
        window_menu.addAction(mdi_action)

        tabbed_action = QAction("&Tabbed Mode", self)
        tabbed_action.triggered.connect(lambda: self.set_display_mode(DisplayMode.TABBED))
        window_menu.addAction(tabbed_action)

        window_menu.addSeparator()

        clone_action = QAction("&Clone Window", self)
        clone_action.triggered.connect(self._clone_window)
        window_menu.addAction(clone_action)


    def _setup_shortcuts(self) -> None:
        """Set up default keyboard shortcuts (e.g., Ctrl+W to close subwindow)."""
        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self._close_active_subwindow)
        self.addAction(close_action)


    def create_toolbar(
        self,
        name: str,
        area: Qt.ToolBarArea = Qt.ToolBarArea.TopToolBarArea
    ) -> QToolBar:
        """Create and add a toolbar to the main window."""
        if name in self._toolbars:
            return self._toolbars[name]

        toolbar = QToolBar(name, self)
        toolbar.setObjectName(name)
        self.addToolBar(area, toolbar)
        self._toolbars[name] = toolbar
        return toolbar


    def get_toolbar(self, name: str) -> QToolBar | None:
        """Get a toolbar by name."""
        return self._toolbars.get(name)


    def remove_toolbar(self, name: str) -> None:
        """Remove a toolbar by name."""
        if not name or name not in self._toolbars:
            return
        toolbar = self._toolbars.pop(name)
        self.removeToolBar(toolbar)
        toolbar.deleteLater()


    def create_dock_widget(
        self,
        name: str,
        widget: QWidget,
        area: Qt.DockWidgetArea = Qt.DockWidgetArea.RightDockWidgetArea,
        allowed_areas: Qt.DockWidgetArea = Qt.DockWidgetArea.AllDockWidgetAreas
    ) -> QDockWidget:
        """Create and add a dock widget to the main window."""
        if name in self._dock_widgets:
            return self._dock_widgets[name]

        dock = QDockWidget(name, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setAllowedAreas(allowed_areas)
        self.addDockWidget(area, dock)
        self._dock_widgets[name] = dock
        return dock


    def get_dock_widget(self, name: str) -> QDockWidget | None:
        """Get a dock widget by name."""
        return self._dock_widgets.get(name)


    def remove_dock_widget(self, name: str) -> None:
        """Remove a dock widget by name."""
        if not name or name not in self._dock_widgets:
            return
        dock = self._dock_widgets.pop(name)
        self.removeDockWidget(dock)
        dock.deleteLater()


    def add_subwindow(self, subwindow: UASSubWindow|None) -> None:
        """Add a subwindow to this main window and make it active."""
        if subwindow is None:
            return
        self._subwindows.append(subwindow)
        self._add_subwindow_to_display(subwindow)
        subwindow.status_message.connect(self._on_status_message)
        self._set_active_subwindow(subwindow)


    def remove_subwindow(self, subwindow: UASSubWindow) -> None:
        """Remove a subwindow from this main window."""
        if not subwindow or not isinstance(subwindow, UASSubWindow) or subwindow not in self._subwindows:
            return
        self._subwindows.remove(subwindow)
        self._remove_subwindow_from_display(subwindow)
        try:
            subwindow.status_message.disconnect(self._on_status_message)
        except (RuntimeError, TypeError):
            pass  # Signal was not connected
        if self._active_subwindow == subwindow:
            self._active_subwindow = None
            if self._subwindows:
                self._set_active_subwindow(self._subwindows[-1])
        subwindow.close()

    def _add_subwindow_to_display(self, subwindow: UASSubWindow) -> None:
        """Add a subwindow to the current display mode container (MDI or tab widget)."""
        if self._display_mode == DisplayMode.MDI and self._mdi_area:
            mdi_sub = QMdiSubWindow()
            mdi_sub.setWidget(subwindow)
            mdi_sub.setWindowTitle(subwindow.title)
            mdi_sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self._mdi_area.addSubWindow(mdi_sub)
            if subwindow.mdi_geometry:
                geo = subwindow.mdi_geometry
                x, y, width, height = geo["x"], geo["y"], geo["width"], geo["height"]

                mdi_rect = self._mdi_area.viewport().rect()
                x, y, width, height = fit_into_geometry(x, y, width, height, mdi_rect)
                mdi_sub.setGeometry(x, y, width, height)
            mdi_sub.show()
        elif self._display_mode == DisplayMode.TABBED and self._tab_widget:
            self._tab_widget.addTab(subwindow, subwindow.title)


    def _remove_subwindow_from_display(self, subwindow: UASSubWindow) -> None:
        """Remove a subwindow from the current display mode container."""
        if self._display_mode == DisplayMode.MDI and self._mdi_area:
            for mdi_sub in self._mdi_area.subWindowList():
                if mdi_sub.widget() == subwindow:
                    mdi_sub.setWidget(None)
                    self._mdi_area.removeSubWindow(mdi_sub)
                    break
        elif self._display_mode == DisplayMode.TABBED and self._tab_widget:
            index = self._tab_widget.indexOf(subwindow)
            if index >= 0:
                self._tab_widget.removeTab(index)


    def _get_mdi_subwindow(self, subwindow: UASSubWindow) -> QMdiSubWindow | None:
        """Get the MDI subwindow container wrapper for a subwindow."""
        if self._display_mode != DisplayMode.MDI or self._mdi_area is None:
            return None
        for mdi_sub in self._mdi_area.subWindowList():
            if mdi_sub.widget() == subwindow:
                return mdi_sub


    def _set_active_subwindow(self, subwindow: UASSubWindow | None) -> None:
        """Set the active subwindow and call on_activate/on_deactivate callbacks."""
        if subwindow is None or not isinstance(subwindow, UASSubWindow) or self._active_subwindow == subwindow:
            return

        if self._active_subwindow:
            self._active_subwindow.on_deactivate()

        self._active_subwindow = subwindow

        subwindow.on_activate()
        if self._display_mode == DisplayMode.MDI and self._mdi_area:
            for mdi_sub in self._mdi_area.subWindowList():
                if mdi_sub.widget() == subwindow:
                    self._mdi_area.setActiveSubWindow(mdi_sub)
                    break
        elif self._display_mode == DisplayMode.TABBED and self._tab_widget:
            index = self._tab_widget.indexOf(subwindow)
            if index >= 0:
                self._tab_widget.setCurrentIndex(index)


    def _on_mdi_subwindow_activated(self, mdi_subwindow: QMdiSubWindow | None) -> None:
        """Handle MDI subwindow activation event from MDI area."""
        if mdi_subwindow is None:
            return
        widget = mdi_subwindow.widget()
        self._set_active_subwindow(widget)


    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change event in tabbed mode."""
        if index >= 0 and self._tab_widget:
            widget = self._tab_widget.widget(index)
            self._set_active_subwindow(widget)


    def _on_tab_close_requested(self, index: int) -> None:
        """Handle tab close request from tabbed mode."""
        if not self._tab_widget or index < 0:
            return
        widget = self._tab_widget.widget(index)
        self.remove_subwindow(widget)


    def _on_status_message(self, message: str) -> None:
        """Handle status message signal from a subwindow and display in status bar."""
        self._status_bar.showMessage(message, 5000)


    def _close_active_subwindow(self) -> None:
        """Close the currently active subwindow (triggered by Ctrl+W shortcut)."""
        self.remove_subwindow(self._active_subwindow)


    def _clone_window(self) -> None:
        """Clone this main window by serializing and creating a new window from state."""
        state = self.serialize()
        state["id"] = str(uuid.uuid4())
        session = SessionManager.get_instance()
        session.create_main_window_from_state(state)


    def _update_mdi_geometries(self) -> None:
        """Update stored MDI geometries from current MDI subwindows for persistence."""
        if self._display_mode != DisplayMode.MDI and self._mdi_area:
            return
        QApplication.processEvents()
        for sw in self._subwindows:
            mdi_sub = self._get_mdi_subwindow(sw)
            if mdi_sub is None:
                continue
            sw.mdi_geometry = mdi_sub.geometry()


    def serialize(self) -> dict[str, Any]:
        """Serialize this main window's state to a dict (override to add custom state)."""
        self._update_mdi_geometries()

        return {
            "type": self.type_name,
            "id": self._id,
            "title": self.windowTitle(),
            "display_mode": self._display_mode.name,
            "geometry": {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
            },
            "subwindows": [sw.serialize() for sw in self._subwindows],
        }


    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore this main window's state from a dict (override to restore custom state)."""
        if "id" in state:
            self._id = state["id"]
        if "title" in state:
            self.setWindowTitle(state["title"])
        if "geometry" in state:
            geo = state["geometry"]
            x, y, width, height = geo["x"], geo["y"], geo["width"], geo["height"]
            screen = QApplication.primaryScreen()
            if screen:
                screen_rect = screen.availableGeometry()
                x, y, width, height = fit_into_geometry(x, y, width, height, screen_rect)

            self.setGeometry(x, y, width, height)
            self.show()
            QApplication.processEvents()
        if "display_mode" in state:
            mode = DisplayMode[state["display_mode"]]
            if mode != self._display_mode:
                self._cleanup_display_mode()
                self._display_mode = mode
                self._setup_display_mode(mode)

        if "subwindows" in state:
            registry = FactoryRegistry.get_instance()
            for sw_state in state["subwindows"]:
                cls = registry.get_subwindow_class(sw_state["type"])
                subwindow = cls.create(self, sw_state)
                self.add_subwindow(subwindow)


    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event, saving session if auto-save enabled and last window."""
        session = SessionManager.get_instance()

        is_last_window = len(session.main_windows) == 1 and self in session.main_windows
        if is_last_window and session.auto_save_enabled:
            session.save()

        for subwindow in self._subwindows[:]:
            subwindow.close()

        session.remove_main_window(self)

        super().closeEvent(event)
