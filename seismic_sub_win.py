import numpy as np
import os
from typing import Any
from enum import Enum

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QMenu, QToolBar, QDialog, QFormLayout, QComboBox, QDialogButtonBox, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from uas import UASSubWindow, UASMainWindow, auto_register

import segyio
from gprpy.toolbox.gprIO_MALA import readMALA
from gs_icon import create_gs_icon
from global_settings import GlobalSettings, UnitSystem

class AxisType(Enum):
    NONE = "None"
    SAMPLE = "Sample"
    TIME = "Time"
    DEPTH = "Depth"
    DISTANCE = "Distance"
    TRACE = "Trace"


axis_type_to_label: dict[AxisType, str] = {
    AxisType.NONE: "None",
    AxisType.SAMPLE: "Sample Number",
    AxisType.TIME: "Time",
    AxisType.DEPTH: "Depth",
    AxisType.DISTANCE: "Distance",
    AxisType.TRACE: "Trace Number",
}

class DisplaySettingsDialog(QDialog):
    """
    Dialog for configuring display axis settings for each axis
    """
    default_settings = {
        'top': AxisType.NONE,
        'bottom': AxisType.DISTANCE,
        'left': AxisType.SAMPLE,
        'right': AxisType.NONE,
        'colormap': 'seismic',
        'flip_colormap': False,
        'colorbar_visible': True,
        'file_name_in_plot': True,
        'top_major_tick': 0.0,
        'top_minor_ticks': 0,
        'bottom_major_tick': 0.0,
        'bottom_minor_ticks': 0,
        'left_major_tick': 0.0,
        'left_minor_ticks': 0,
        'right_major_tick': 0.0,
        'right_minor_ticks': 0,
    }
    colormap_options = ["seismic", "gray", "viridis", "plasma", "RdBu", "hot", "coolwarm", "jet"]
    vertical_options = [AxisType.NONE, AxisType.SAMPLE, AxisType.TIME, AxisType.DEPTH]
    horizontal_options = [AxisType.NONE, AxisType.DISTANCE, AxisType.TRACE]

    @staticmethod
    def _enum_to_string_list(enum_list: list[AxisType]) -> list[str]:
        """Convert list of AxisType enums to their string values."""
        return [axis_type.value for axis_type in enum_list]


    def __init__(self, parent: 'SeismicSubWindow', current_settings: dict[str, Any]|None = None):
        super().__init__(parent)
        self.setWindowTitle("Display Settings")
        self._parent = parent
        self._old_settings = dict(current_settings) if current_settings else dict(self.default_settings)

        if current_settings is None:
            current_settings = self.default_settings
        else:
            current_settings = {**self.default_settings, **current_settings}

        # Create layout
        layout = QVBoxLayout(self)

        # Create grid layout for axes settings (table format without titles)
        axes_group = QGroupBox("Axes properties")
        grid_layout = QGridLayout()

        # Row 0: Top axis
        grid_layout.addWidget(QLabel("Top:"), 0, 0)
        self.top_combo = QComboBox()
        self.top_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.top_combo.setCurrentText(current_settings.get('top', DisplaySettingsDialog.default_settings['top']).value)
        self.top_combo.currentIndexChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.top_combo, 0, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 0, 2)
        self.top_major_tick = QSpinBox()
        self.top_major_tick.setRange(0, 10000)
        self.top_major_tick.setValue(current_settings['top_major_tick'])
        self.top_major_tick.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.top_major_tick, 0, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 0, 4)
        self.top_minor_ticks = QSpinBox()
        self.top_minor_ticks.setRange(0, 100)
        self.top_minor_ticks.setValue(current_settings['top_minor_ticks'])
        self.top_minor_ticks.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.top_minor_ticks, 0, 5)

        # Row 1: Bottom axis
        grid_layout.addWidget(QLabel("Bottom:"), 1, 0)
        self.bottom_combo = QComboBox()
        self.bottom_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.bottom_combo.setCurrentText(current_settings.get('bottom', DisplaySettingsDialog.default_settings['bottom']).value)
        self.bottom_combo.currentIndexChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.bottom_combo, 1, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 1, 2)
        self.bottom_major_tick = QSpinBox()
        self.bottom_major_tick.setRange(0, 10000)
        self.bottom_major_tick.setValue(current_settings['bottom_major_tick'])
        self.bottom_major_tick.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.bottom_major_tick, 1, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 1, 4)
        self.bottom_minor_ticks = QSpinBox()
        self.bottom_minor_ticks.setRange(0, 100)
        self.bottom_minor_ticks.setValue(current_settings['bottom_minor_ticks'])
        self.bottom_minor_ticks.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.bottom_minor_ticks, 1, 5)

        # Row 2: Left axis
        grid_layout.addWidget(QLabel("Left:"), 2, 0)
        self.left_combo = QComboBox()
        self.left_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.left_combo.setCurrentText(current_settings.get('left', DisplaySettingsDialog.default_settings['left']).value)
        self.left_combo.currentIndexChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.left_combo, 2, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 2, 2)
        self.left_major_tick = QSpinBox()
        self.left_major_tick.setRange(0, 10000)
        self.left_major_tick.setValue(current_settings['left_major_tick'])
        self.left_major_tick.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.left_major_tick, 2, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 2, 4)
        self.left_minor_ticks = QSpinBox()
        self.left_minor_ticks.setRange(0, 100)
        self.left_minor_ticks.setValue(current_settings['left_minor_ticks'])
        self.left_minor_ticks.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.left_minor_ticks, 2, 5)

        # Row 3: Right axis
        grid_layout.addWidget(QLabel("Right:"), 3, 0)
        self.right_combo = QComboBox()
        self.right_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.right_combo.setCurrentText(current_settings.get('right', DisplaySettingsDialog.default_settings['right']).value)
        self.right_combo.currentIndexChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.right_combo, 3, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 3, 2)
        self.right_major_tick = QSpinBox()
        self.right_major_tick.setRange(0, 10000)
        self.right_major_tick.setValue(current_settings['right_major_tick'])
        self.right_major_tick.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.right_major_tick, 3, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 3, 4)
        self.right_minor_ticks = QSpinBox()
        self.right_minor_ticks.setRange(0, 100)
        self.right_minor_ticks.setValue(current_settings['right_minor_ticks'])
        self.right_minor_ticks.valueChanged.connect(self._on_setting_changed)
        grid_layout.addWidget(self.right_minor_ticks, 3, 5)

        axes_group.setLayout(grid_layout)
        layout.addWidget(axes_group)

        # Colormap selection
        colormap_group = QGroupBox("Colormap properties")
        
        colormap_layout = QHBoxLayout()

        # Colorbar visible checkbox
        show_colorbar_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        self.colorbar_visible_checkbox = QCheckBox("Show Colorbar")
        self.colorbar_visible_checkbox.setChecked(current_settings.get('colorbar_visible', DisplaySettingsDialog.default_settings['colorbar_visible']))
        self.colorbar_visible_checkbox.stateChanged.connect(self._on_setting_changed)
        show_colorbar_layout.addWidget(self.colorbar_visible_checkbox)
        colormap_layout.addLayout(show_colorbar_layout)

        # Common colormaps
        colorscheme_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        colorscheme_layout.addWidget(QLabel("Color scheme:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(DisplaySettingsDialog.colormap_options)
        self.colormap_combo.setCurrentText(current_settings.get('colormap', DisplaySettingsDialog.default_settings['colormap']))
        self.colormap_combo.currentIndexChanged.connect(self._on_setting_changed)
        colorscheme_layout.addWidget(self.colormap_combo)
        colormap_layout.addLayout(colorscheme_layout)

        # Flip colormap checkbox
        flip_checkbox_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        self.flip_colormap_checkbox = QCheckBox("Flip colormap")
        self.flip_colormap_checkbox.setChecked(current_settings.get('flip_colormap', DisplaySettingsDialog.default_settings['flip_colormap']))
        self.flip_colormap_checkbox.stateChanged.connect(self._on_setting_changed)
        flip_checkbox_layout.addWidget(self.flip_colormap_checkbox)
        colormap_layout.addLayout(flip_checkbox_layout)

        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)

        # File name in plot checkbox
        self.file_name_in_plot_checkbox = QCheckBox("Show file name in plot")
        self.file_name_in_plot_checkbox.setChecked(current_settings.get('file_name_in_plot', DisplaySettingsDialog.default_settings['file_name_in_plot']))
        self.file_name_in_plot_checkbox.stateChanged.connect(self._on_setting_changed)
        layout.addWidget(self.file_name_in_plot_checkbox)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


    def _on_setting_changed(self):
        """Called when any setting changes - update parent display immediately."""
        new_settings = self.get_settings()
        old_colorbar = self._old_settings.get('colorbar_visible', True)
        new_colorbar = new_settings.get('colorbar_visible', True)

        # Update parent's settings
        self._parent._display_settings = new_settings

        # If colorbar visibility changed, need to recreate subplots
        if old_colorbar != new_colorbar:
            self._parent._recreate_subplots()
        else:
            # Otherwise just re-render
            self._parent.canvas_render()

        # Update old settings for next comparison
        self._old_settings = dict(new_settings)

    def reject(self):
        """Restore old settings when dialog is cancelled."""
        self._parent._display_settings = self._old_settings
        self._parent.canvas_render()
        super().reject()

    def get_settings(self):
        """Return the selected settings as a dictionary."""
        return {
            'top': AxisType(self.top_combo.currentText()),
            'bottom': AxisType(self.bottom_combo.currentText()),
            'left': AxisType(self.left_combo.currentText()),
            'right': AxisType(self.right_combo.currentText()),
            'colormap': self.colormap_combo.currentText(),
            'flip_colormap': self.flip_colormap_checkbox.isChecked(),
            'colorbar_visible': self.colorbar_visible_checkbox.isChecked(),
            'file_name_in_plot': self.file_name_in_plot_checkbox.isChecked(),
            'top_major_tick': self.top_major_tick.value(),
            'top_minor_ticks': self.top_minor_ticks.value(),
            'bottom_major_tick': self.bottom_major_tick.value(),
            'bottom_minor_ticks': self.bottom_minor_ticks.value(),
            'left_major_tick': self.left_major_tick.value(),
            'left_minor_ticks': self.left_minor_ticks.value(),
            'right_major_tick': self.right_major_tick.value(),
            'right_minor_ticks': self.right_minor_ticks.value(),
        }


@auto_register
class SeismicSubWindow(UASSubWindow):
    """
    Subwindow for displaying seismic SEGY data.

    Purpose:
        Displays seismic data from SEGY or MALA files using matplotlib with a seismic
        colormap. Supports color scale adjustment and session persistence.

    Flow:
        1. Initialize with empty canvas
        2. Load SEGY or MALA file via load_file() method
        3. Data is displayed with seismic colormap
        4. State (filename, color scale) is serialized for session persistence
    """

    type_name = "seismic"


    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        # Call parent init (will call empty on_create)
        super().__init__(main_window, parent)

        self._data: np.ndarray | None = None
        self._filename: str = ""
        self._amplitude_min: float = 0.0
        self._amplitude_max: float = 0.0
        self._colorbar: plt.Colorbar | None = None
        self._image: plt_image.AxesImage | None = None
        self._colorbar_indicator: Line2D | None = None

        # Metadata fields
        self._trace_coords: np.ndarray | None = None
        self._trace_cumulative_distances: np.ndarray | None = None
        self._sample_interval_seconds: float = 1.0
        self._sample_min_seconds: float = 0.0
        self._z_display_units: str = ""
        self._z_display_value_factor: float = 1.0
        self._is_depth: bool = False

        # Zoom state
        self._zoom_active: bool = False
        self._zoom_id: int | None = None
        self._press_event = None
        self._sticky_vertical_edges: bool = False
        self._sticky_horizontal_edges: bool = False

        # Display settings
        self._display_settings: dict[str, Any] = dict(DisplaySettingsDialog.default_settings)

        # Create shared zoom menu
        self._zoom_menu = None
        self._create_zoom_menu()

        self.title = "Seismic View"
        self._fig = Figure(figsize=(10, 6))

        self.create_subplots()
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Create matplotlib navigation toolbar (for pan/zoom)
        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
        # Hide coordinate display in toolbar
        self._nav_toolbar.set_message = lambda x: None

        # Create custom toolbar for zoom toggle
        self._toolbar = QToolBar("Seismic Toolbar", self)
        self._setup_toolbar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._nav_toolbar)
        layout.addWidget(self._canvas)

        self.setMinimumSize(400, 300)

        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect canvas hover events to propagate to main window
        self._canvas.mpl_connect("axes_leave_event", self._axes_leave_event)
        self._canvas.mpl_connect("motion_notify_event", self._motion_notify_event)

        # Set custom window icon
        self.setWindowIcon(create_gs_icon())

        # Connect resize event to update layout
        self._canvas.mpl_connect('resize_event', self._on_resize)

        # Register as listener for global settings changes
        GlobalSettings.add_listener(self._on_global_settings_changed)


    def _get_unit_label(self, axis_type: AxisType) -> str:
        if axis_type in [AxisType.TIME, AxisType.DEPTH]:
            return self._z_display_units
        return ""

    def _get_axis_geometry(self, axis_type: AxisType) -> tuple:
        if axis_type in [AxisType.TIME, AxisType.DEPTH]:
            return (-self._sample_min_seconds*self._z_display_value_factor, self._sample_interval_seconds*self._z_display_value_factor, self._data.shape[0])
        if axis_type in [AxisType.DISTANCE]:
            nx = self._data.shape[1]
            dx = self._trace_cumulative_distances[-1] / (nx - 1)
            return (0, dx*GlobalSettings.display_length_factor, nx)
        return (0, 1, self._data.shape[1])

    def _show_error(self, title: str, message: str) -> None:
        """Show error message by rendering it on the matplotlib canvas.

        This method:
        1. Prints the error to terminal (so it's visible in logs)
        2. Renders error message as text on the matplotlib canvas
        3. No dialog needed - error is visible in screenshots automatically
        """
        # Print to terminal
        print(f"ERROR: {title} - {message}", flush=True)

        # Clear the axes and display error message on canvas
        self._axes.clear()
        self._axes.set_xlim(0, 1)
        self._axes.set_ylim(0, 1)
        self._axes.axis('off')  # Hide axes

        # Add error icon/symbol at top
        self._axes.text(0.5, 0.75, '⚠️',
                       ha='center', va='center',
                       fontsize=80, color='red',
                       transform=self._axes.transAxes)

        # Add error title
        self._axes.text(0.5, 0.55, title,
                       ha='center', va='center',
                       fontsize=20, weight='bold', color='darkred',
                       transform=self._axes.transAxes)

        # Add error message (wrap text if needed)
        self._axes.text(0.5, 0.35, message,
                       ha='center', va='center',
                       fontsize=14, color='black',
                       wrap=True,
                       transform=self._axes.transAxes)

        # Set background color to light red
        self._axes.set_facecolor('#ffebee')

        # Redraw canvas
        self._canvas.draw()


    def _setup_toolbar(self) -> None:
        """Set up the toolbar with zoom toggle button."""
        self._zoom_action = QAction("🔍 Zoom", self)
        self._zoom_action.setCheckable(True)
        self._zoom_action.setChecked(False)
        self._zoom_action.setToolTip("Toggle zoom mode (Left-click drag to zoom in, Right-click to zoom out)")
        self._zoom_action.toggled.connect(self._toggle_zoom)
        self._toolbar.addAction(self._zoom_action)

        # Make toolbar visible and set proper size
        self._toolbar.setVisible(True)
        self._toolbar.setMovable(False)

        # Style the toolbar to make checked state more visible
        self._toolbar.setStyleSheet("""
            QToolBar QToolButton:checked {
                border: 2px solid #0078d4;
                background-color: transparent;
            }
        """)

        # Install event filter to catch right-clicks on toolbar
        self._toolbar.setContextMenuPolicy(Qt.CustomContextMenu)
        self._toolbar.customContextMenuRequested.connect(self._show_zoom_context_menu)


    def _create_zoom_menu(self) -> None:
        """Create the shared zoom submenu."""
        self._zoom_menu = QMenu("Zoom", self)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self._zoom_out)
        self._zoom_menu.addAction(reset_zoom_action)

        # Add separator
        self._zoom_menu.addSeparator()

        # Sticky Vertical Edges (checkable)
        self._sticky_vertical_action = QAction("Sticky Vertical Edges", self)
        self._sticky_vertical_action.setCheckable(True)
        self._sticky_vertical_action.setChecked(self._sticky_vertical_edges)
        self._sticky_vertical_action.toggled.connect(self._toggle_sticky_vertical)
        self._zoom_menu.addAction(self._sticky_vertical_action)

        # Sticky Horizontal Edges (checkable)
        self._sticky_horizontal_action = QAction("Sticky Horizontal Edges", self)
        self._sticky_horizontal_action.setCheckable(True)
        self._sticky_horizontal_action.setChecked(self._sticky_horizontal_edges)
        self._sticky_horizontal_action.toggled.connect(self._toggle_sticky_horizontal)
        self._zoom_menu.addAction(self._sticky_horizontal_action)


    def _show_zoom_context_menu(self, position) -> None:
        """Show context menu for zoom tool with reset option."""
        context_menu = QMenu(self)

        # Add shared zoom submenu
        context_menu.addMenu(self._zoom_menu)

        # Show menu at cursor position
        context_menu.exec(self._toolbar.mapToGlobal(position))


    def _toggle_zoom(self, checked: bool) -> None:
        """Toggle zoom mode on/off."""
        self._zoom_active = checked
        if checked:
            # Activate zoom mode
            self._activate_zoom()
        else:
            # Deactivate zoom mode
            self._deactivate_zoom()


    def _activate_zoom(self) -> None:
        """Activate zoom mode by connecting mouse events."""
        self._zoom_id = self._canvas.mpl_connect('button_press_event', self._on_zoom_press)
        self._canvas.setCursor(Qt.CrossCursor)


    def _deactivate_zoom(self) -> None:
        """Deactivate zoom mode by disconnecting mouse events."""
        if self._zoom_id is not None:
            self._canvas.mpl_disconnect(self._zoom_id)
            self._zoom_id = None
        self._canvas.unsetCursor()


    def _on_zoom_press(self, event) -> None:
        """Handle mouse press for zoom rectangle selection."""
        if event.inaxes != self._axes:
            return

        # Only handle left click for zoom
        if event.button == 1:  # Left click - start zoom rectangle
            self._press_event = event
            self._release_id = self._canvas.mpl_connect('button_release_event', self._on_zoom_release)
            self._motion_id = self._canvas.mpl_connect('motion_notify_event', self._on_zoom_motion)
            self._rect = None


    def _on_zoom_motion(self, event) -> None:
        """Draw zoom rectangle during mouse drag."""
        if self._press_event is None:
            return

        # Allow dragging outside axes but use last valid coordinates
        if event.xdata is None or event.ydata is None:
            return

        # Remove old rectangle
        if hasattr(self, '_rect') and self._rect is not None:
            self._rect.remove()

        # Draw new rectangle from corner to corner
        x0, y0 = self._press_event.xdata, self._press_event.ydata
        x1, y1 = event.xdata, event.ydata

        # Rectangle spans from one corner to opposite corner
        self._rect = self._axes.add_patch(
            plt.Rectangle((min(x0, x1), min(y0, y1)),
                         abs(x1 - x0), abs(y1 - y0),
                         fill=False, edgecolor='red', linewidth=2, linestyle='--')
        )
        self._canvas.draw_idle()


    def _on_zoom_release(self, event) -> None:
        """Complete zoom rectangle selection and apply zoom."""
        if self._press_event is None:
            return

        # Disconnect motion and release events
        self._canvas.mpl_disconnect(self._release_id)
        self._canvas.mpl_disconnect(self._motion_id)

        # Get coordinates (use last valid position if released outside axes)
        x0, y0 = self._press_event.xdata, self._press_event.ydata

        # If released outside axes, use the rectangle's last position
        if event.xdata is not None and event.ydata is not None:
            x1, y1 = event.xdata, event.ydata
        elif hasattr(self, '_rect') and self._rect is not None:
            # Use last rectangle position
            bbox = self._rect.get_bbox()
            x1, y1 = bbox.x1, bbox.y1
        else:
            x1, y1 = x0, y0

        # Remove rectangle - it disappears on mouse release
        if hasattr(self, '_rect') and self._rect is not None:
            self._rect.remove()
            self._rect = None
            self._canvas.draw_idle()

        # Apply zoom if rectangle is large enough (at least a few pixels)
        x_mid = int(round((x0 + x1) / 2))
        y_mid = int(round((y0 + y1) / 2))
        dx = max(5, int(round(abs(x1 - x0)/2))) + 0.5
        dy = max(5, int(round(abs(y1 - y0)/2))) + 0.5
        self._axes.set_xlim(x_mid - dx, x_mid + dx)
        self._axes.set_ylim(y_mid + dy, y_mid - dy)  # Inverted for image coordinates
        self._canvas.draw_idle()

        self._press_event = None


    def _zoom_out(self) -> None:
        """Zoom out to show full data extent."""
        if self._data is not None:
            self._axes.set_xlim(-0.5, self._data.shape[1] - 0.5)
            self._axes.set_ylim(self._data.shape[0] - 0.5, -0.5)  # Inverted for image coordinates
            self._canvas.draw_idle()


    def _toggle_sticky_vertical(self, checked: bool) -> None:
        """Toggle sticky vertical edges mode."""
        self._sticky_vertical_edges = checked


    def _toggle_sticky_horizontal(self, checked: bool) -> None:
        """Toggle sticky horizontal edges mode."""
        self._sticky_horizontal_edges = checked


    @staticmethod
    def create_from_load_file(main_window: UASMainWindow) -> None:
        """Create a new seismic subwindow from a loaded file."""
        subwindow = SeismicSubWindow(main_window)
        if subwindow.load_file_dialog():
            main_window.add_subwindow(subwindow)
        else:
            subwindow.deleteLater()


    @property
    def canvas(self) -> FigureCanvasQTAgg:
        """Get the seismic canvas."""
        return self._canvas


    @property
    def axes(self) -> plt.Axes | None:
        """Get the matplotlib axes."""
        return self._axes


    @property
    def data(self) -> np.ndarray | None:
        """Get the loaded seismic data."""
        return self._data


    @property
    def filename(self) -> str:
        """Get the loaded filename."""
        return self._filename


    def _motion_notify_event(self, event) -> None:
        """Handle mouse motion to show amplitude indicator on colorbar."""
        hover_info = None
        if event is not None and event.xdata is not None and event.ydata is not None:
            # Get pixel coordinates
            hover_info = self.get_hover_info(event.xdata, event.ydata)

        amplitude = hover_info.get('amplitude', None) if hover_info is not None else None
        self._update_colorbar_indicator(amplitude)
        self._main_window.on_subwindow_hover(hover_info)


    def _axes_leave_event(self, event) -> None:
        """Handle canvas leave event and clear status bar."""
        self._update_colorbar_indicator(None)
        self._main_window.on_subwindow_hover({})


    def _update_colorbar_indicator(self, amplitude: float|None) -> None:
        """Update the horizontal indicator line on the colorbar."""
        self.remove_colorbar_indicator()

        if self._image is None or self._colorbar is None or amplitude is None:
            return
        # If amplitude is None, use the middle of the colorbar
        if abs(self._amplitude_min - self._amplitude_max) < 1e-9:
            norm_value = 0.5
        else:
            norm_value = (amplitude - self._amplitude_min) / (self._amplitude_max - self._amplitude_min)
            norm_value = max(0.0, min(1.0, norm_value))  # Clamp to [0, 1]
        cmap = self._image.get_cmap()
        rgba = cmap(norm_value)
        inv_color = (1.0 - rgba[0], 1.0 - rgba[1], 1.0 - rgba[2])
        self._colorbar_indicator = self._colorbar.ax.axhline(y=amplitude, color=inv_color, linewidth=2, alpha=1.0)
        self._canvas.draw_idle()


    def get_hover_info(self, x: float, y: float) -> dict[str, Any] | None:
        """
        Calculate hover information for given pixel coordinates.

        Args:
            x: Trace number (horizontal pixel coordinate)
            y: Sample number (vertical pixel coordinate)

        Returns:
            Dictionary with hover info, or None if out of bounds
        """
        if self._data is None:
            return None

        ix, iy = int(round(x)), int(round(y))

        # Check bounds
        if not (0 <= iy < self._data.shape[0] and 0 <= ix < self._data.shape[1]):
            return None
        z_value = (iy * self._sample_interval_seconds - self._sample_min_seconds) * self._z_display_value_factor
        z_units = self._z_display_units
        if self._trace_cumulative_distances is not None:
            if ix == len(self._trace_cumulative_distances) - 1:
                distance = self._trace_cumulative_distances[-1]
            else:
                dx = x - ix
                distance = self._trace_cumulative_distances[ix] * (1-dx) + self._trace_cumulative_distances[ix+1] * dx
            distance *= GlobalSettings.display_length_factor
        else:
            distance = None

        hover_info = {
            'trace_number': ix,
            'sample_number': iy,
            'z_value': z_value,
            'z_units': z_units,
            'is_depth': self._is_depth,  # True = depth, False = time
            'amplitude': self._data[iy, ix],
            'distance': distance,
        }

        return hover_info


    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including loaded file and color scale."""
        state = super().serialize()
        state['filename'] = self._filename

        # Convert AxisType enums to strings for JSON serialization
        serialized_settings = dict(self._display_settings)
        for key in ['top', 'bottom', 'left', 'right']:
            if key in serialized_settings and isinstance(serialized_settings[key], AxisType):
                serialized_settings[key] = serialized_settings[key].value

        state['display_settings'] = serialized_settings
        return state


    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including loaded file and color scale."""
        super().deserialize(state)
        if 'display_settings' in state:
            settings = state['display_settings']
            # Convert string values back to AxisType enums
            for key in ['top', 'bottom', 'left', 'right']:
                if key not in settings or not isinstance(settings[key], str):
                    settings[key] = DisplaySettingsDialog.default_settings[key]
                    continue
                try:
                    settings[key] = AxisType(settings[key])
                except ValueError:
                    settings[key] = DisplaySettingsDialog.default_settings[key]
            self._display_settings = settings
        self.load_file(state.get("filename", ""))


    def _show_context_menu(self, position) -> None:
        """Show context menu on right-click."""
        context_menu = QMenu(self)

        # Load action (single option for all formats)
        load_action = QAction("Load...", self)
        load_action.triggered.connect(self.load_file_dialog)
        context_menu.addAction(load_action)

        # Save submenu
        save_menu = QMenu("Save as...", self)

        save_segy_action = QAction("As SEGY File...", self)
        save_segy_action.triggered.connect(self._save_segy)
        save_menu.addAction(save_segy_action)

        save_rd3_action = QAction("As MALA rd3 File...", self)
        save_rd3_action.triggered.connect(self._save_rd3)
        save_menu.addAction(save_rd3_action)

        save_rd7_action = QAction("As MALA rd7 File...", self)
        save_rd7_action.triggered.connect(self._save_rd7)
        save_menu.addAction(save_rd7_action)

        context_menu.addMenu(save_menu)

        # Display Settings action
        display_settings_action = QAction("Display Settings...", self)
        display_settings_action.triggered.connect(self._show_display_settings)
        context_menu.addAction(display_settings_action)

        # Add shared zoom submenu
        context_menu.addMenu(self._zoom_menu)

        # Show menu at cursor position
        context_menu.exec(self.mapToGlobal(position))


    def load_file_dialog(self) -> bool:
        """Load a seismic file (SEGY, rd3, or rd7)."""
        # Use directory of current file as default, or empty string if no file loaded
        default_dir = ""
        if self.filename:
            default_dir = os.path.dirname(self.filename)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Seismic File",
            default_dir,
            "Seismic Files (*.sgy *.segy *.rd3 *.rd7);;SEGY Files (*.sgy *.segy);;MALA Files (*.rd3 *.rd7);;All Files (*)",
        )
        if not filename:
            return False
        return self.load_file(filename)


    def load_file(self, filename: str) -> bool:
        """Load a seismic file (SEGY, rd3, or rd7) into this subwindow."""
        if not filename or not os.path.exists(filename) or not os.path.isfile(filename):
            self._show_error("Error", f"Invalid file: {filename}")
            return False
        
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in ['.rd3', '.rd7']:
            file_type = file_ext[1:]
        else:
            file_type = 'sgy'
        success = self.canvas_load_file(filename, file_type)
        if success:
            self.title = os.path.basename(filename)
            self.update_status(f"Loaded: {filename}")
        return success


    def canvas_load_file(self, filename: str, file_type: str) -> bool:
        if file_type not in ['sgy', 'rd3', 'rd7']:
            self._show_error("Error", f"Invalid file type: {file_type}")
            return False

        if file_type == 'sgy':
            ret, error_message = self._load_segy_data(filename)
        else:
            ret, error_message = self._load_mala_data(filename)

        if not ret:
            self._show_error("Error", f"Failed to load file: {filename}\n{error_message}")
            self.remove_colorbar_indicator()
            self.remove_colorbar()
            self._trace_coords = None
            self._trace_cumulative_distances = None
            self._distance_unit = "trace"
            self._sample_interval_seconds = 1.0
            self._sample_min_seconds = 0.0
            self._is_depth = False
            self._amplitude_min = 0.0
            self._amplitude_max = 0.0
            self._image = None
            self._filename = ""
            return False

        self._filename = filename
        self._amplitude_min = np.min(self._data)
        self._amplitude_max = np.max(self._data)
        z_range_seconds = self._data.shape[0] * self._sample_interval_seconds
        if self._sample_min_seconds < 0 or self._sample_min_seconds >= z_range_seconds:
            # default to 10% of the data range
            self._sample_min_seconds = self._data.shape[0] * self._sample_interval_seconds * 0.1
        self._sample_min_seconds = round(self._sample_min_seconds / self._sample_interval_seconds) * self._sample_interval_seconds
        if z_range_seconds < 0.01:
            # to nano seconds
            self._z_display_units = "ns"
            self._z_display_value_factor = 1_000_000_000.0
        elif z_range_seconds < 10.0:
            # to milliseconds
            self._z_display_units = "ms"
            self._z_display_value_factor = 1000.0
        else:
            # keep in seconds
            self._z_display_units = "s"
            self._z_display_value_factor = 1.0

        # Calculate cumulative distances along the survey line
        # Distance from trace i-1 to trace i for each trace
        assert self._trace_coords is not None
        assert self._trace_coords.shape[1] == 2
        trace_distances = np.sqrt(np.sum(np.diff(self._trace_coords, axis=0)**2, axis=1))
        # Cumulative sum with 0.0 prepended for first trace
        self._trace_cumulative_distances = np.cumsum(np.concatenate([[0.0], trace_distances]))

        self.canvas_render()
        return ret


    def _load_segy_data(self, filename: str) -> tuple[bool, str]:
        try:
            with segyio.open(filename, "r", ignore_geometry=True) as f:
                self._data = f.trace.raw[:].T
                if self._data is None or self._data.size == 0:
                    return False, "No data found in SEGY file"
                # Extract metadata from SEGY file
                # Sample interval from binary header (bytes 3217-3218, in microseconds per SEG-Y standard)
                dt_us = f.bin[segyio.BinField.Interval]
                if dt_us <= 0:
                    return False, "Sample interval is not set in SEGY file"
                self._sample_interval_seconds = dt_us / 1_000_000.0  # Convert microseconds to seconds

                # Check if depth or time (assume time by default for SEGY)
                self._is_depth = False

                # Try to extract trace coordinates from trace headers
                num_traces = len(f.trace)
                coords = np.full((num_traces, 2), fill_value=np.nan)

                segy_coord_unit_map = {
                    1: "length", # meters or feet
                    2: "arcsec", # seconds of arc
                    3: "deg", # decimal degrees
                    4: "DMS", # degrees, minutes, seconds
                }
                segy_measurement_system_map = {
                    1: UnitSystem.MKS,
                    2: UnitSystem.IMPERIAL,
                }
                # Read measurement system from binary header (bytes 3255-3256)
                measurement_system = f.bin[segyio.BinField.MeasurementSystem]
                file_unit_system = segy_measurement_system_map.get(measurement_system, UnitSystem.MKS)
                convert_2_mks = UnitSystem.convert_length_factor(file_unit_system, UnitSystem.MKS)
                time_delays_seconds = np.full(num_traces, fill_value=np.nan)
                count_valid_coords = 0
                for i in range(num_traces):
                    trace_header = f.header[i]
                    # DelayRecordingTime from trace header (bytes 109-110, in milliseconds per SEG-Y standard)
                    delay_ms = trace_header[segyio.TraceField.DelayRecordingTime]
                    if delay_ms > 0:
                        time_delays_seconds[i] = delay_ms / 1000.0  # Convert milliseconds to seconds
                    x, y = trace_header[segyio.TraceField.CDP_X], trace_header[segyio.TraceField.CDP_Y]
                    if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):
                        continue
                    count_valid_coords += 1
                    scalar = trace_header[segyio.TraceField.SourceGroupScalar]
                    if scalar < 0:
                        x /= abs(scalar)
                        y /= abs(scalar)
                    elif scalar > 0:
                        x *= float(scalar)
                        y *= float(scalar)
                    else:
                        pass
                    coord_units = trace_header[segyio.TraceField.CoordinateUnits]
                    if coord_units in segy_coord_unit_map:
                        if segy_coord_unit_map[coord_units] == "length":
                            x *= convert_2_mks
                            y *= convert_2_mks
                            pass
                        pass
                    coords[i] = [x, y]

                if not count_valid_coords:
                    return False, "No trace coordinates found in SEGY file"
                # interpolate nan values in coords
                arrange_indices = np.arange(num_traces)
                is_nan_coords = np.isnan(coords[:, 0]) | np.isnan(coords[:, 1])
                coords[is_nan_coords] = np.interp(arrange_indices[is_nan_coords], arrange_indices[~is_nan_coords], coords[~is_nan_coords])
                self._trace_coords = coords
                is_nan_time_delays = np.isnan(time_delays_seconds)
                time_delays_seconds[is_nan_time_delays] = np.interp(arrange_indices[is_nan_time_delays], arrange_indices[~is_nan_time_delays], time_delays_seconds[~is_nan_time_delays])
                self._trace_time_delays_seconds = time_delays_seconds
                self._sample_min_seconds = np.median(time_delays_seconds)
        except Exception as e:
            return False, f"Error loading SEGY file: {e}"

        return True, ""


    def _load_mala_data(self, filename: str) -> tuple[bool, str]:
        try:
            file_base, _ = os.path.splitext(filename)
            data, info = readMALA(file_base)
            self._data = np.array(data)

            # Extract metadata from MALA header
            # Calculate sample interval (dt) from TIMEWINDOW and SAMPLES
            # TIMEWINDOW is in nanoseconds
            timewindow_ns = float(info.get('TIMEWINDOW', 0))
            samples = self._data.shape[0]
            if timewindow_ns > 0 and samples > 1:
                dt_ns = timewindow_ns / (samples - 1)
                self._sample_interval_seconds = dt_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            else:
                self._sample_interval_seconds = 1.0

            # MALA is typically time-based (GPR data)
            self._is_depth = False

            # Extract signal position (time zero offset) from MALA header
            # SIGNAL POSITION is in nanoseconds
            signal_position_ns = float(info.get('SIGNAL POSITION', 0))
            self._sample_min_seconds = signal_position_ns / 1_000_000_000.0  # Convert nanoseconds to seconds

            # Distance interval from MALA header
            distance_interval = float(info.get('DISTANCE INTERVAL', 0))
            if distance_interval <= 0:
                return False, "Distance interval is not set in MALA file"
            num_traces = self._data.shape[1]
            # Create linear coordinates based on distance interval
            x_coords = np.arange(num_traces) * distance_interval
            self._trace_coords = np.column_stack([x_coords, np.zeros(num_traces)])
            self._trace_time_delays_seconds = np.full(num_traces, fill_value=self._sample_min_seconds)
        except Exception as e:
            return False, f"Error loading MALA file: {e}"

        return True, ""


    def canvas_render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._data is None:
            return

        # Save current zoom state before clearing
        xlim = self._axes.get_xlim()
        ylim = self._axes.get_ylim()

        self.remove_colorbar()
        self.remove_colorbar_indicator()

        self._axes.clear()

        # Restore zoom or set initial limits for new data
        if xlim == (0.0, 1.0):  # Default uninitialized state
            self._axes.set_xlim(-0.5, self._data.shape[1] - 0.5)
            self._axes.set_ylim(self._data.shape[0] - 0.5, -0.5)
        else:
            self._axes.set_xlim(xlim)
            self._axes.set_ylim(ylim)

        self._apply_display_settings()
        self._adjust_layout_with_fixed_margins()

        # Force canvas draw to initialize colorbar axis transforms
        self._canvas.draw()
        self._canvas.draw_idle()


    def _on_resize(self, event) -> None:
        """Handle canvas resize event to maintain fixed pixel margins."""
        if self._data is not None:
            self._adjust_layout_with_fixed_margins()


    def _adjust_layout_with_fixed_margins(self) -> None:
        """Adjust figure layout with fixed pixel margins from global settings."""

        # Get figure size in pixels
        width_px = self._fig.get_figwidth() * self._fig.dpi
        height_px = self._fig.get_figheight() * self._fig.dpi

        # Convert pixels to proportions
        horizontal_axes_margin_px = GlobalSettings.margins_px['horizontal_axes']
        vertical_axes_margin_px = GlobalSettings.margins_px['vertical_axes']
        base_vertical_margin_px = GlobalSettings.margins_px['base_vertical']
        base_horizontal_margin_px = GlobalSettings.margins_px['base_horizontal']

        top_margin_px = base_vertical_margin_px
        if self._display_settings.get('top', AxisType.NONE) != AxisType.NONE:
            top_margin_px += horizontal_axes_margin_px
        if self._display_settings.get('file_name_in_plot', True):
            top_margin_px += base_vertical_margin_px

        bottom_margin_px = base_vertical_margin_px
        if self._display_settings.get('bottom', AxisType.NONE) != AxisType.NONE:
            bottom_margin_px += horizontal_axes_margin_px

        bottom = bottom_margin_px / height_px
        height = 1.0 - (top_margin_px + bottom_margin_px) / height_px

        left_image_margin_px = base_horizontal_margin_px
        if self._display_settings.get('left', AxisType.NONE) != AxisType.NONE:
            left_image_margin_px += vertical_axes_margin_px

        right_image_margin_px = base_horizontal_margin_px
        if self._display_settings.get('right', AxisType.NONE) != AxisType.NONE:
            right_image_margin_px += vertical_axes_margin_px

        if self.is_colorbar_axes_visible():
            colorbar_width_px = GlobalSettings.margins_px['colorbar_width']
            colorbar_right_margin_px = vertical_axes_margin_px + base_horizontal_margin_px
            colorbar_left_margin_px = colorbar_width_px + colorbar_right_margin_px
            colorbar_left = 1 - colorbar_left_margin_px / width_px
            colorbar_width = colorbar_width_px / width_px
            self._colorbar_axes.set_position([colorbar_left, bottom, colorbar_width, height])
            right_image_margin_px += colorbar_left_margin_px


        image_left = left_image_margin_px / width_px
        image_width = 1.0 - (left_image_margin_px + right_image_margin_px) / width_px
        self._axes.set_position([image_left, bottom, image_width, height])
        self._canvas.draw_idle()


    def _on_global_settings_changed(self) -> None:
        """Callback when global settings change - update the layout."""
        if self._data is None:
            return
        self._adjust_layout_with_fixed_margins()
        self._canvas.draw_idle()


    def _on_local_settings_changed(self) -> None:
        """Callback when local settings change - update the layout."""
        if self._data is None:
            return
        self._apply_display_settings()
        self._adjust_layout_with_fixed_margins()
        self._canvas.draw_idle()


    def _recreate_subplots(self) -> None:
        """Recreate subplots with updated parameters."""

        # Clear existing subplots
        self._fig.clear()

        self.create_subplots()
        # Re-render if we have data
        if self._data is not None:
            self.canvas_render()


    def create_subplots(self) -> None:
        # Note: We'll set margins and wspace later via _adjust_layout_with_fixed_margins
        self.remove_colorbar_axes()
        if self._display_settings.get('colorbar_visible', True):
            self._axes = self._fig.add_subplot(1, 2, 1)
            self._colorbar_axes = self._fig.add_subplot(1, 2, 2)
        else:
            self._axes = self._fig.add_subplot(1, 1, 1)
            self._colorbar_axes = None


    def remove_colorbar_axes(self) -> None:
        #check if colorbar_axes is defined, if so, remove it
        if not hasattr(self, '_colorbar_axes'):
            self._colorbar_axes = None
            return
        if self._colorbar_axes is None:
            return
        self.remove_colorbar()
        if self._colorbar_axes.figure is self._fig and self._colorbar_axes in self._fig.axes:
            self._colorbar_axes.remove()
        self._colorbar_axes = None


    def remove_colorbar(self) -> None:
        if self._colorbar is None:
            return
        self.remove_colorbar_indicator()

        # Only remove if the colorbar's axes is still in the figure
        if self._colorbar.ax.figure is self._fig and self._colorbar.ax in self._fig.axes:
            self._colorbar.remove()
        self._colorbar = None


    def remove_colorbar_indicator(self) -> None:
        if self._colorbar_indicator is None:
            return        
        self._colorbar_indicator.set_visible(False)
        self._colorbar_indicator = None


    def closeEvent(self, event) -> None:
        """Handle window close event - unregister from global settings."""
        GlobalSettings.remove_listener(self._on_global_settings_changed)
        super().closeEvent(event)


    def _show_display_settings(self) -> None:
        """Show the display settings dialog with live updates (modeless)."""
        # Check if dialog already exists and is visible
        if hasattr(self, '_settings_dialog') and self._settings_dialog is not None:
            # Bring existing dialog to front
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return

        # Create and show modeless dialog
        self._settings_dialog = DisplaySettingsDialog(self, self._display_settings)
        self._settings_dialog.finished.connect(self._on_settings_dialog_closed)
        self._settings_dialog.show()


    def _on_settings_dialog_closed(self):
        """Clean up when settings dialog is closed."""
        self._settings_dialog = None


    def _apply_display_settings(self) -> None:
        """Apply the display settings to the axes."""
        if self._data is None:
            return

        colormap = self._display_settings.get('colormap', 'seismic')
        flip_colormap = self._display_settings.get('flip_colormap', False)
        file_name_in_plot = self._display_settings.get('file_name_in_plot', True)
        # Add '_r' suffix to flip the colormap
        if flip_colormap:
            colormap = colormap + '_r'
        self._image = self._axes.imshow(self._data, aspect="auto", cmap=colormap, vmin=self._amplitude_min, vmax=self._amplitude_max)
        if self.is_colorbar_axes_visible():
            self._colorbar = self._fig.colorbar(self._image, cax=self._colorbar_axes, label="Amplitude [mV]")
            colorbar_ticks = list(self._colorbar.get_ticks())
            # assuming the colorbar ticks are already sorted, add min and max if not already present
            while colorbar_ticks and colorbar_ticks[0] <= self._amplitude_min:
                colorbar_ticks.pop(0)
            while colorbar_ticks and colorbar_ticks[-1] >= self._amplitude_max:
                colorbar_ticks.pop(-1)
            colorbar_ticks = [self._amplitude_min] + colorbar_ticks + [self._amplitude_max]
            self._colorbar.set_ticks(colorbar_ticks)

        if file_name_in_plot:
            self._axes.set_title(os.path.basename(self._filename))

        # Get current settings
        top = self._display_settings.get('top', AxisType.NONE)
        bottom = self._display_settings.get('bottom', AxisType.DISTANCE)
        left = self._display_settings.get('left', AxisType.SAMPLE)
        right = self._display_settings.get('right', AxisType.NONE)

        # remove top and bottom labels
        self._axes.set_xlabel(None)
        self._axes.set_ylabel(None)
        self._axes.secondary_xaxis('top').set_visible(False)

        # Apply top axis
        if top == AxisType.NONE:
            self._axes.xaxis.set_tick_params(top=False, labeltop=False)
            self._axes.secondary_xaxis('top').set_visible(False)
        else:
            ax2 = self._axes.secondary_xaxis('top')
            ax2.set_visible(True)
            ax2.tick_params(axis='x', top=True, labeltop=True)
            self._apply_tick_settings(
                ax2.xaxis,
                top,
                self._display_settings.get('top_major_tick', 0.0),
                self._display_settings.get('top_minor_ticks', 0)
            )

        # Apply bottom axis
        if bottom == AxisType.NONE:
            self._axes.xaxis.set_tick_params(bottom=False, labelbottom=False)
        else:
            self._axes.xaxis.set_tick_params(bottom=True, labelbottom=True)
            self._apply_tick_settings(
                self._axes.xaxis,
                bottom,
                self._display_settings.get('bottom_major_tick', 0.0),
                self._display_settings.get('bottom_minor_ticks', 0)
            )

        # Apply left axis
        if left == AxisType.NONE:
            self._axes.yaxis.set_tick_params(left=False, labelleft=False)
        else:
            self._axes.yaxis.set_tick_params(left=True, labelleft=True)
            self._apply_tick_settings(
                self._axes.yaxis,
                left,
                self._display_settings.get('left_major_tick', 0.0),
                self._display_settings.get('left_minor_ticks', 0)
            )
            # Apply tick settings for left axis

        # Apply right axis
        if right == AxisType.NONE:
            self._axes.yaxis.set_tick_params(right=False, labelright=False)
            self._axes.secondary_yaxis('right').set_visible(False)
        else:
            ax2 = self._axes.secondary_yaxis('right')
            ax2.set_visible(True)
            ax2.tick_params(axis='y', right=True, labelright=True)
            self._apply_tick_settings(
                ax2.yaxis,
                right,
                self._display_settings.get('right_major_tick', 0.0),
                self._display_settings.get('right_minor_ticks', 0)
            )


    def is_colorbar_axes_visible(self) -> bool:
        colorbar_visible = self._display_settings.get('colorbar_visible', True) # must check with default value True
        if not colorbar_visible:
            self.remove_colorbar_axes()
            return False
            
        if self._colorbar_axes is not None and self._colorbar_axes.figure is not self._fig:
            self.remove_colorbar_axes()
        if self._colorbar_axes is None:
            if len(self._fig.axes) == 1: # if there is only one subplot, add a new subplot for the colorbar
                self._colorbar_axes = self._fig.add_subplot(1, 2, 2)
            else:
                self._colorbar_axes = self._fig.axes[1]
        return True


    def _apply_tick_settings(self, axis: plt.Axes, axis_type: AxisType, major_tick_distance: float, minor_ticks_per_major: int) -> None:
        """
        Apply tick settings to an axis with a reference offset.

        Args:
            axis: matplotlib axis object (e.g., self._axes.xaxis or self._axes.yaxis)
            axis_type: AxisType for the axis
            axis_geometry: Tuple containing the axis minimum, step, and number of samples
            major_tick_distance: Distance between major ticks (0 = auto)
            minor_ticks_per_major: Number of minor ticks between major ticks (0 = none)
            offset: Reference point for tick alignment (default: 0.0)
        """
        label = axis_type_to_label[axis_type]
        axis_unit_label = self._get_unit_label(axis_type)
        if axis_unit_label:
            label = f"{label} [{axis_unit_label}]"
        axis.set_label_text(label)

        def n_ticks(view_val, distance) -> int:
            # Calculate tick multiplier for exact multiples of distance
            # Uses int() truncation to only include ticks at 0, ±distance, ±2*distance, etc.
            # Example: view_val=-10, distance=25 -> returns 0 (no tick at -25 since -10 > -25)
            #          view_val=-30, distance=25 -> returns -1 (includes tick at -25)
            return int(abs(view_val) / distance) * int(np.sign(view_val))

        # For distance axis, convert pixel indices to distance values in tick labels
        if axis_type == AxisType.DISTANCE:
            def format_distance(x, pos):
                # x is pixel index, convert to distance
                ix = int(x+0.5)
                ix = max(0, min(ix, len(self._trace_cumulative_distances) - 1))
                distance = self._trace_cumulative_distances[ix] * GlobalSettings.display_length_factor
                return f'{distance:.3g}'
            axis.set_major_formatter(FuncFormatter(format_distance))

        axis_min, axis_step, axis_num_samples = self._get_axis_geometry(axis_type)
        minor_ticks_per_major = min(minor_ticks_per_major, axis_num_samples)
        axis_max = axis_min + axis_step * (axis_num_samples - 1)
        if major_tick_distance <= axis_step:
            major_tick_distance = axis_step
            minor_ticks_per_major = 0
        if major_tick_distance >= axis_step*(axis_num_samples-1):
            major_tick_distance = axis_step*(axis_num_samples-1)
            minor_ticks_per_major = max(min(10,axis_num_samples), minor_ticks_per_major)
        # Calculate tick positions in display units
        n_min = n_ticks(axis_min, major_tick_distance)
        n_max = n_ticks(axis_max, major_tick_distance)
        if n_min >= n_max:
            # Use automatic tick placement with custom formatter for distance axis
            axis.set_major_locator(plt.AutoLocator())
            axis.set_minor_locator(AutoMinorLocator())
            return

        major_tick_values = np.arange(n_min, n_max + 1) * major_tick_distance

        # Convert display unit positions to data coordinates (pixel indices)
        major_tick_positions = (major_tick_values - axis_min) / axis_step

        # Set ticks at data coordinate positions with display unit labels
        axis.set_ticks(major_tick_positions)
        axis.set_ticklabels([f'{val:.3g}' for val in major_tick_values])

        if minor_ticks_per_major < 2:
            axis.set_minor_locator(plt.NullLocator())
            return

        # Calculate minor tick positions
        minor_tick_distance = major_tick_distance / minor_ticks_per_major
        n_min_minor = n_ticks(axis_min, minor_tick_distance)
        n_max_minor = n_ticks(axis_max, minor_tick_distance)
        minor_tick_values = [i * minor_tick_distance for i in range(n_min_minor, n_max_minor + 1)
                            if i % minor_ticks_per_major != 0]

        # Convert to data coordinates
        minor_tick_positions = [(val - axis_min) / axis_step for val in minor_tick_values]
        axis.set_ticks(minor_tick_positions, minor=True)


    def _save_segy(self) -> None:
        self.save_file('sgy')

    def _save_rd3(self) -> None:
        self.save_file('rd3')

    def _save_rd7(self) -> None:
        """Save current view as MALA rd7 file with retry logic."""
        self.save_file('rd7')


    def save_file(self, file_type: str) -> None:
        """
        Save current data to a file.

        """
        if self._data is None:
            self._show_error("Error", "No data to save")
            return

        while True:
            # Create default filename by concatenating current filename with .file_type
            default_name = ""
            if self.filename:
                default_name = self.filename + '.' + file_type
            label = ''
            filter = ''
            if file_type == 'sgy':
                label = 'SEGY'
                filter = '(*.sgy *.segy)'
            elif file_type == 'rd3':
                label = 'MALA rd3'
                filter = '(*.rd3)'
            elif file_type == 'rd7':
                label = 'MALA rd7'
                filter = '(*.rd7)'
            else:
                self._show_error("Error", f"Invalid file type: {file_type}")
                return

            filename, _ = QFileDialog.getSaveFileName(self, f"Save as {label} File", default_name, f"{label} Files ({filter});;All Files (*)")
            if not filename:
                # User cancelled
                return

            try:
                if file_type == 'sgy':
                    # Create a minimal SEGY file with current data
                    spec = segyio.spec()
                    spec.samples = range(self._data.shape[0])
                    spec.tracecount = self._data.shape[1]
                    spec.format = 1  # 4-byte IBM float

                    with segyio.create(filename, spec) as f:
                        for i, trace in enumerate(self._data.T):
                            f.trace[i] = trace
                else: # rd3 or rd7
                    file_base, _ = os.path.splitext(filename)
                    data_file = file_base + '.' + file_type

                    if file_type == 'rd3':
                        data_normalized = self._data / np.max(np.abs(self._data))
                        data_int16 = (data_normalized * 32767).astype(np.int16)

                        # Save binary data
                        data_int16.T.tofile(data_file)
                    else:
                        data_float32 = self._data.astype(np.float32)
                        data_float32.T.tofile(data_file)
                QMessageBox.information(self, "Success", f"Saved to {filename}")
                break
            except Exception as e:
                action = self._show_save_error_dialog(str(e), label)
                if action == 'cancel':
                    break
                if action == 'change_format':
                    # Show save format menu
                    self._show_save_format_menu()



    def _show_save_error_dialog(self, error_msg: str, format_type: str) -> str:
        """
        Show error dialog with retry options.

        Returns: 'retry', 'change_format', or 'cancel'
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Save Error")
        msg_box.setText(f"Failed to save {format_type} file")
        msg_box.setInformativeText(error_msg)

        retry_btn = msg_box.addButton("Retry", QMessageBox.AcceptRole)
        change_format_btn = msg_box.addButton("Save As Different Format", QMessageBox.ActionRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.exec_()

        clicked = msg_box.clickedButton()
        if clicked == retry_btn:
            return 'retry'
        elif clicked == change_format_btn:
            return 'change_format'
        else:
            return 'cancel'



    def _show_save_format_menu(self) -> None:
        """Show a menu to select save format."""
        menu = QMenu(self)

        segy_action = QAction("SEGY Format", self)
        segy_action.triggered.connect(self._save_segy)
        menu.addAction(segy_action)

        rd3_action = QAction("MALA rd3 Format", self)
        rd3_action.triggered.connect(self._save_rd3)
        menu.addAction(rd3_action)

        rd7_action = QAction("MALA rd7 Format", self)
        rd7_action.triggered.connect(self._save_rd7)
        menu.addAction(rd7_action)

        # Show menu at cursor
        menu.exec(self.mapToGlobal(self.mapFromGlobal(self.cursor().pos())))

