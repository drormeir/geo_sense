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

    @staticmethod
    def convert_length_factor(source: 'UnitSystem', target: 'UnitSystem') -> float:
        if source == target:
            return 1.0
        return 0.3048 if target == UnitSystem.IMPERIAL else 3.28084


class GlobalSettings(QDialog):
    """
    Singleton class for managing global application settings.

    Combines both the settings data model and the dialog UI.
    All windows and subwindows access the same instance to ensure
    consistent settings across the application.

    Uses observer pattern to notify listeners when settings change.
    """

    _instance = None
    _initialized = False
    margin_prefix: str = 'margin_px_'
    margins_px: dict[str, int] = {
        'base_vertical': 30,
        'horizontal_axes': 30,
        'base_horizontal': 10,
        'vertical_axes': 70,
        'colorbar_width': 30,
    }

    display_unit_system: UnitSystem = UnitSystem.MKS
    display_unit_system_prefix: str = 'unit_system'
    display_unit_system_options: list[str] = [UnitSystem.MKS.value, UnitSystem.IMPERIAL.value]
    display_length_factor: float = UnitSystem.convert_length_factor(UnitSystem.MKS, display_unit_system)
    display_length_unit: str = "m" if display_unit_system == UnitSystem.MKS else "ft"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalSettings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize default settings and dialog UI if not already initialized."""
        if self._initialized:
            return

        # Observer pattern - list of listener callbacks
        self._listeners: list[Callable[[], None]] = []

        # Flag to track if UI has been initialized
        self._ui_initialized: bool = False

        self._initialized = True

    @classmethod
    def _get_instance(cls) -> 'GlobalSettings':
        """Get the singleton instance of GlobalSettings (internal use only)."""
        return cls()


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
        """
        Serialize settings to a dictionary for session persistence.

        Returns:
            Dictionary containing all settings.
        """
        ret = {GlobalSettings.margin_prefix + key: value for key, value in GlobalSettings.margins_px.items()}
        ret[GlobalSettings.display_unit_system_prefix] = GlobalSettings.display_unit_system.value
        return ret


    @classmethod
    def deserialize(cls, state: dict[str, Any]) -> None:
        """
        Restore settings from a dictionary.

        Args:
            state: Dictionary containing serialized settings.
        """
        """
        Restore settings from a dictionary.

        Args:
            state: Dictionary containing serialized settings.
        """
        for key, value in state.items():
            if key.startswith(GlobalSettings.margin_prefix):
                GlobalSettings.margins_px[key[len(GlobalSettings.margin_prefix):]] = value
        GlobalSettings.set_display_unit_system(state.get(GlobalSettings.display_unit_system_prefix, UnitSystem.MKS.value))


    @classmethod
    def set_display_unit_system(cls, value: UnitSystem|str) -> None:
        try:
            value = UnitSystem(value)
        except Exception as e:
            value = UnitSystem.MKS
            print(f"Error converting unit system: {e}\nUsing default unit system: {UnitSystem.MKS.value}")
        cls.display_unit_system = value
        cls.display_length_factor = UnitSystem.convert_length_factor(cls.display_unit_system, UnitSystem.MKS)
        cls.display_length_unit = "m" if cls.display_unit_system == UnitSystem.MKS else "ft"



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

        self._margin_spins: dict[str, QSpinBox] = {}
        keys = list(GlobalSettings.margins_px.keys())
        titles = [key.replace('_', ' ').capitalize() for key in keys]
        max_length = max(len(title) for title in titles)
        titles = {key: title.ljust(max_length) for key, title in zip(keys, titles)}
        for key in keys:
            spin = QSpinBox()
            spin.setRange(0, 200)
            spin.setValue(GlobalSettings.margins_px[key])
            margins_layout.addRow(titles[key] + ":", spin)
            self._margin_spins[key] = spin


        margins_group.setLayout(margins_layout)
        layout.addWidget(margins_group)

        # Unit system group
        units_group = QGroupBox("Unit System")
        units_layout = QFormLayout()

        self._unit_combo = QComboBox()
        self._unit_combo.addItems(GlobalSettings.display_unit_system_options)
        units_layout.addRow("Display units:", self._unit_combo)

        units_group.setLayout(units_layout)
        layout.addWidget(units_group)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accepted)
        button_box.rejected.connect(self._on_rejected)
        layout.addWidget(button_box)

        # Connect all spin boxes to live update
        self._connect_live_updates()


    def _connect_live_updates(self) -> None:
        """Connect all spin boxes to trigger live updates on value change."""
        for spin in self._margin_spins.values():
            spin.valueChanged.connect(self._apply_settings_from_widgets)
        self._unit_combo.currentTextChanged.connect(self._apply_settings_from_widgets)


    def _load_settings_to_widgets(self) -> None:
        """Load current settings values into dialog widgets."""
        for key, value in GlobalSettings.margins_px.items():
            self._margin_spins[key].setValue(value)
        self._unit_combo.setCurrentText(GlobalSettings.display_unit_system.value)


    def _on_accepted(self) -> None:
        """Handle OK button - apply settings and notify listeners."""
        self._apply_settings_from_widgets()
        self.hide()


    def _on_rejected(self) -> None:
        """Handle Cancel button - just hide without applying."""
        self.hide()


    def _apply_settings_from_widgets(self) -> None:
        """Apply widget values to settings data."""
        if not self.isVisible():
            return
        for key, spin in self._margin_spins.items():
            GlobalSettings.margins_px[key] = spin.value()
        GlobalSettings.set_display_unit_system(self._unit_combo.currentText())
        """Notify all registered listeners that settings have changed."""
        for callback in self._listeners:
            try:
                callback()
            except Exception as e:
                print(f"Error notifying listener: {e}")

