"""
Global Settings for GeoSense Application

Provides a singleton class to manage application-wide settings including
plot margins and unit systems. Settings are accessible from any window
and persist across sessions.
"""

from typing import Any, Callable
from enum import Enum

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QWidget,
    QMenu,
)
from PySide6.QtGui import QAction

from uas import SessionManager


class UnitSystem(Enum):
    """Unit system enumeration."""
    MKS = "MKS"  # Meter-Kilogram-Second
    IMPERIAL = "Imperial"  # Foot-Pound-Second


class GlobalSettings(QDialog):
    """
    Singleton class for managing global application settings.

    Combines both the settings data model and the dialog UI.
    All windows and subwindows access the same instance to ensure
    consistent settings across the application.

    Uses observer pattern to notify listeners when settings change.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalSettings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize default settings and dialog UI if not already initialized."""
        if self._initialized:
            return

        # Plot margin settings (in pixels)
        self.left_margin_px: int = 60
        # Right margins stack outward from the image:
        self.right_margin_image_axis_px: int = 60      # Space for right image axis labels (closest to image)
        self.right_margin_colorbar_px: int = 60        # Space for colorbar (middle)
        self.right_margin_colorbar_label_px: int = 50  # Space for "Amplitude" label (outermost)
        self.top_margin_px: int = 40
        self.bottom_margin_px: int = 50

        # Unit system
        self.unit_system: UnitSystem = UnitSystem.MKS

        # Observer pattern - list of listener callbacks
        self._listeners: list[Callable[[], None]] = []

        # Flag to track if UI has been initialized
        self._ui_initialized: bool = False

        self._initialized = True

    @classmethod
    def _get_instance(cls) -> 'GlobalSettings':
        """Get the singleton instance of GlobalSettings (internal use only)."""
        return cls()

    # ========== Public Class Methods (Namespace-style API) ==========

    @classmethod
    def get_left_margin_px(cls) -> int:
        """Get left margin in pixels."""
        return cls._get_instance().left_margin_px

    @classmethod
    def get_right_margin_image_axis_px(cls) -> int:
        """Get right image axis margin in pixels."""
        return cls._get_instance().right_margin_image_axis_px

    @classmethod
    def get_right_margin_colorbar_px(cls) -> int:
        """Get right colorbar margin in pixels."""
        return cls._get_instance().right_margin_colorbar_px

    @classmethod
    def get_right_margin_colorbar_label_px(cls) -> int:
        """Get right colorbar label margin in pixels."""
        return cls._get_instance().right_margin_colorbar_label_px

    @classmethod
    def get_top_margin_px(cls) -> int:
        """Get top margin in pixels."""
        return cls._get_instance().top_margin_px

    @classmethod
    def get_bottom_margin_px(cls) -> int:
        """Get bottom margin in pixels."""
        return cls._get_instance().bottom_margin_px

    @classmethod
    def get_unit_system(cls) -> UnitSystem:
        """Get the current unit system."""
        return cls._get_instance().unit_system

    @classmethod
    def get_total_right_margin(cls) -> int:
        """
        Get total right margin (sum of all right components stacked outward from image).

        Stack order from image outward:
        1. Image axis labels (closest to image)
        2. Colorbar
        3. Colorbar label (outermost)

        Returns:
            Total right margin in pixels.
        """
        instance = cls._get_instance()
        return (instance.right_margin_image_axis_px +
                instance.right_margin_colorbar_px +
                instance.right_margin_colorbar_label_px)

    @classmethod
    def add_listener(cls, callback: Callable[[], None]) -> None:
        """
        Register a callback to be notified when settings change.

        Args:
            callback: Function to call when settings are updated
        """
        instance = cls._get_instance()
        if callback not in instance._listeners:
            instance._listeners.append(callback)

    @classmethod
    def remove_listener(cls, callback: Callable[[], None]) -> None:
        """
        Unregister a callback.

        Args:
            callback: Function to remove from listeners
        """
        instance = cls._get_instance()
        if callback in instance._listeners:
            instance._listeners.remove(callback)

    @classmethod
    def show_dialog(cls) -> None:
        """Show the global settings dialog."""
        instance = cls._get_instance()
        instance._ensure_ui_initialized()
        instance._load_settings_to_widgets()
        instance.show()
        instance.raise_()
        instance.activateWindow()

    @classmethod
    def create_action(cls, settings_menu: QMenu, parent: QWidget, label: str = "&Global Settings...") -> QAction:
        """
        Create and add a QAction to a menu that opens the global settings dialog.

        Args:
            settings_menu: The menu to add the action to
            parent: Parent widget for the action
            label: Text label for the action (default: "&Global Settings...")

        Returns:
            QAction that was created and added to the menu
        """
        action = QAction(label, parent)
        action.triggered.connect(cls.show_dialog)
        settings_menu.addAction(action)
        return action

    @classmethod
    def serialize(cls) -> dict[str, Any]:
        """
        Serialize settings to a dictionary for session persistence.

        Returns:
            Dictionary containing all settings.
        """
        return cls._get_instance()._serialize_instance()

    @classmethod
    def deserialize(cls, state: dict[str, Any]) -> None:
        """
        Restore settings from a dictionary.

        Args:
            state: Dictionary containing serialized settings.
        """
        cls._get_instance()._deserialize_instance(state)

    @classmethod
    def register(cls) -> None:
        """
        Register GlobalSettings with the SessionManager for persistence.

        This should be called once from the main() function before the application runs.
        It registers the GlobalSettings singleton as a session plugin so settings are
        automatically saved and restored across sessions.
        """
        session = SessionManager.get_instance()
        session.register_plugin("global_settings", cls._get_instance())

    # ========== Internal Instance Methods ==========

    def _ensure_ui_initialized(self) -> None:
        """Ensure the dialog UI is initialized (lazy initialization)."""
        if self._ui_initialized:
            return

        # Initialize QDialog - must be called after QApplication exists
        super().__init__(None)  # No parent - dialog is independent
        self.setWindowTitle("Global Settings")

        self._create_ui()
        self._ui_initialized = True

    def _create_ui(self) -> None:
        """Create the dialog UI widgets."""
        # Create layout
        layout = QVBoxLayout(self)

        # Margins group
        margins_group = QGroupBox("Plot Margins (pixels)")
        margins_layout = QFormLayout()

        self._left_spin = QSpinBox()
        self._left_spin.setRange(0, 200)
        margins_layout.addRow("Left margin:", self._left_spin)

        self._right_image_axis_spin = QSpinBox()
        self._right_image_axis_spin.setRange(0, 200)
        margins_layout.addRow("Right image axis:", self._right_image_axis_spin)

        self._right_colorbar_spin = QSpinBox()
        self._right_colorbar_spin.setRange(0, 200)
        margins_layout.addRow("Right colorbar:", self._right_colorbar_spin)

        self._right_colorbar_label_spin = QSpinBox()
        self._right_colorbar_label_spin.setRange(0, 200)
        margins_layout.addRow("Right colorbar label:", self._right_colorbar_label_spin)

        self._top_spin = QSpinBox()
        self._top_spin.setRange(0, 200)
        margins_layout.addRow("Top margin:", self._top_spin)

        self._bottom_spin = QSpinBox()
        self._bottom_spin.setRange(0, 200)
        margins_layout.addRow("Bottom margin:", self._bottom_spin)

        margins_group.setLayout(margins_layout)
        layout.addWidget(margins_group)

        # Unit system group
        units_group = QGroupBox("Unit System")
        units_layout = QFormLayout()

        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["MKS", "Imperial"])
        units_layout.addRow("Display units:", self._unit_combo)

        units_group.setLayout(units_layout)
        layout.addWidget(units_group)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self._on_rejected)
        layout.addWidget(button_box)

    def _serialize_instance(self) -> dict[str, Any]:
        """
        Serialize settings to a dictionary for session persistence.

        Returns:
            Dictionary containing all settings.
        """
        return {
            'left_margin_px': self.left_margin_px,
            'right_margin_image_axis_px': self.right_margin_image_axis_px,
            'right_margin_colorbar_px': self.right_margin_colorbar_px,
            'right_margin_colorbar_label_px': self.right_margin_colorbar_label_px,
            'top_margin_px': self.top_margin_px,
            'bottom_margin_px': self.bottom_margin_px,
            'unit_system': self.unit_system.value,
        }

    def _deserialize_instance(self, state: dict[str, Any]) -> None:
        """
        Restore settings from a dictionary.

        Args:
            state: Dictionary containing serialized settings.
        """
        self.left_margin_px = state.get('left_margin_px', 60)
        self.right_margin_image_axis_px = state.get('right_margin_image_axis_px', 60)
        self.right_margin_colorbar_px = state.get('right_margin_colorbar_px', 60)
        self.right_margin_colorbar_label_px = state.get('right_margin_colorbar_label_px', 50)
        self.top_margin_px = state.get('top_margin_px', 40)
        self.bottom_margin_px = state.get('bottom_margin_px', 50)

        unit_str = state.get('unit_system', 'MKS')
        self.unit_system = UnitSystem(unit_str)

    def _notify_listeners(self) -> None:
        """Notify all registered listeners that settings have changed."""
        for callback in self._listeners:
            try:
                callback()
            except Exception as e:
                print(f"Error notifying listener: {e}")

    def _load_settings_to_widgets(self) -> None:
        """Load current settings values into dialog widgets."""
        self._left_spin.setValue(self.left_margin_px)
        self._right_image_axis_spin.setValue(self.right_margin_image_axis_px)
        self._right_colorbar_spin.setValue(self.right_margin_colorbar_px)
        self._right_colorbar_label_spin.setValue(self.right_margin_colorbar_label_px)
        self._top_spin.setValue(self.top_margin_px)
        self._bottom_spin.setValue(self.bottom_margin_px)
        self._unit_combo.setCurrentText(self.unit_system.value)

    def _on_accepted(self) -> None:
        """Handle OK button - apply settings and notify listeners."""
        self._apply_settings_from_widgets()
        self._notify_listeners()
        self.hide()

    def _on_rejected(self) -> None:
        """Handle Cancel button - just hide without applying."""
        self.hide()

    def _apply_settings_from_widgets(self) -> None:
        """Apply widget values to settings data."""
        self.left_margin_px = self._left_spin.value()
        self.right_margin_image_axis_px = self._right_image_axis_spin.value()
        self.right_margin_colorbar_px = self._right_colorbar_spin.value()
        self.right_margin_colorbar_label_px = self._right_colorbar_label_spin.value()
        self.top_margin_px = self._top_spin.value()
        self.bottom_margin_px = self._bottom_spin.value()
        self.unit_system = UnitSystem(self._unit_combo.currentText())
