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
from matplotlib.ticker import AutoMinorLocator
from uas import (
    UASSubWindow,
    UASMainWindow,
    auto_register, 
    format_value,
    simple_interpolation,
    interpolate_inplace_nan_values,
)

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

# Speed of light (for GPR/electromagnetic waves)
C_VACUUM = 299_792_458.0  # m/s
C_VACUUM_METERS_PER_NANOSECOND = C_VACUUM * 1e-9
# GPR electromagnetic wave velocities
GPR_VELOCITY_AIR = C_VACUUM / 1.000293          # ~299,704,645 m/s
GPR_VELOCITY_WATER = C_VACUUM / 9.0             # ~33,310,273 m/s (at radio frequencies)
GPR_VELOCITY_SAND = C_VACUUM / 2.0              # ~149,896,229 m/s (dry sand)
GPR_GROUND_VELOCITY_DEFAULT = C_VACUUM / 3.0    # ~100,000,000 m/s or 0.1 m/ns (typical soil, εr ≈ 9)

# Seismic/acoustic wave velocities (P-wave)
SEISMIC_P_VELOCITY_AIR = 343.0             # m/s (at 20°C)
SEISMIC_P_VELOCITY_WATER = 1500.0          # m/s
SEISMIC_P_VELOCITY_SAND = 400.0            # m/s (dry sand, approximate)
SEISMIC_P_VELOCITY_SANDSTONE = 2000.0      # m/s (saturated, approximate)



class DisplaySettingsDialog(QDialog):
    """
    Dialog for configuring display axis settings for each axis
    """
    default_settings = {
        'top': AxisType.DISTANCE,
        'bottom': AxisType.TRACE,
        'left': AxisType.TIME,
        'right': AxisType.DEPTH,
        'colormap': 'seismic',
        'flip_colormap': False,
        'colorbar_visible': True,
        'file_name_in_plot': True,
        'top_major_tick': 10.0,
        'top_minor_ticks': 2,
        'bottom_major_tick': 10.0,
        'bottom_minor_ticks': 2,
        'left_major_tick': 10.0,
        'left_minor_ticks': 2,
        'right_major_tick': 10.0,
        'right_minor_ticks': 2,
        'air_velocity_m_per_s': GPR_VELOCITY_AIR,
        'ground_velocity_m_per_s': GPR_GROUND_VELOCITY_DEFAULT,
        'ind_sample_time_first_arrival': 30,
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

        if current_settings is None:
            self._old_settings = dict(self.default_settings)
        else:
            self._old_settings = {**self.default_settings, **current_settings}
        self._create_layout(self._old_settings)


    def _create_layout(self, current_settings: dict[str, Any]) -> None:
        # Create layout
        layout = QVBoxLayout(self)

        # File name in plot checkbox
        self.file_name_in_plot_checkbox = QCheckBox("Show file name in plot")
        self.file_name_in_plot_checkbox.setChecked(current_settings['file_name_in_plot'])
        self.file_name_in_plot_checkbox.stateChanged.connect(self._on_image_axes_settings_changed)
        layout.addWidget(self.file_name_in_plot_checkbox)

        # Create grid layout for axes settings (table format without titles)
        image_axes_group = QGroupBox("Image's axes properties")
        grid_layout = QGridLayout()

        # Row 0: Top axis
        grid_layout.addWidget(QLabel("Top:"), 0, 0)
        self.top_combo = QComboBox()
        self.top_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.top_combo.setCurrentText(current_settings['top'].value)
        self.top_combo.currentIndexChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.top_combo, 0, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 0, 2)
        self.top_major_tick = QSpinBox()
        self.top_major_tick.setRange(0, 10000)
        self.top_major_tick.setValue(current_settings['top_major_tick'])
        self.top_major_tick.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.top_major_tick, 0, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 0, 4)
        self.top_minor_ticks = QSpinBox()
        self.top_minor_ticks.setRange(0, 100)
        self.top_minor_ticks.setValue(current_settings['top_minor_ticks'])
        self.top_minor_ticks.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.top_minor_ticks, 0, 5)

        # Row 1: Bottom axis
        grid_layout.addWidget(QLabel("Bottom:"), 1, 0)
        self.bottom_combo = QComboBox()
        self.bottom_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.bottom_combo.setCurrentText(current_settings['bottom'].value)
        self.bottom_combo.currentIndexChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.bottom_combo, 1, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 1, 2)
        self.bottom_major_tick = QSpinBox()
        self.bottom_major_tick.setRange(0, 10000)
        self.bottom_major_tick.setValue(current_settings['bottom_major_tick'])
        self.bottom_major_tick.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.bottom_major_tick, 1, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 1, 4)
        self.bottom_minor_ticks = QSpinBox()
        self.bottom_minor_ticks.setRange(0, 100)
        self.bottom_minor_ticks.setValue(current_settings['bottom_minor_ticks'])
        self.bottom_minor_ticks.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.bottom_minor_ticks, 1, 5)

        # Row 2: Left axis
        grid_layout.addWidget(QLabel("Left:"), 2, 0)
        self.left_combo = QComboBox()
        self.left_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.left_combo.setCurrentText(current_settings['left'].value)
        self.left_combo.currentIndexChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.left_combo, 2, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 2, 2)
        self.left_major_tick = QSpinBox()
        self.left_major_tick.setRange(0, 10000)
        self.left_major_tick.setValue(current_settings['left_major_tick'])
        self.left_major_tick.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.left_major_tick, 2, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 2, 4)
        self.left_minor_ticks = QSpinBox()
        self.left_minor_ticks.setRange(0, 100)
        self.left_minor_ticks.setValue(current_settings['left_minor_ticks'])
        self.left_minor_ticks.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.left_minor_ticks, 2, 5)

        # Row 3: Right axis
        grid_layout.addWidget(QLabel("Right:"), 3, 0)
        self.right_combo = QComboBox()
        self.right_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.right_combo.setCurrentText(current_settings['right'].value)
        self.right_combo.currentIndexChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.right_combo, 3, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 3, 2)
        self.right_major_tick = QSpinBox()
        self.right_major_tick.setRange(0, 10000)
        self.right_major_tick.setValue(current_settings['right_major_tick'])
        self.right_major_tick.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.right_major_tick, 3, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 3, 4)
        self.right_minor_ticks = QSpinBox()
        self.right_minor_ticks.setRange(0, 100)
        self.right_minor_ticks.setValue(current_settings['right_minor_ticks'])
        self.right_minor_ticks.valueChanged.connect(self._on_image_axes_settings_changed)
        grid_layout.addWidget(self.right_minor_ticks, 3, 5)

        image_axes_group.setLayout(grid_layout)
        layout.addWidget(image_axes_group)

        # Colormap selection
        colormap_group = QGroupBox("Colormap properties")        
        colormap_layout = QHBoxLayout()

        # Colorbar visible checkbox
        show_colorbar_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        self.colorbar_visible_checkbox = QCheckBox("Show Colorbar")
        self.colorbar_visible_checkbox.setChecked(current_settings['colorbar_visible'])
        self.colorbar_visible_checkbox.stateChanged.connect(self._show_hide_colorbar)
        show_colorbar_layout.addWidget(self.colorbar_visible_checkbox)
        colormap_layout.addLayout(show_colorbar_layout)

        # Common colormaps
        colorscheme_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        colorscheme_layout.addWidget(QLabel("Color scheme:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(DisplaySettingsDialog.colormap_options)
        self.colormap_combo.setCurrentText(current_settings['colormap'])
        self.colormap_combo.currentIndexChanged.connect(self._update_colormap)
        colorscheme_layout.addWidget(self.colormap_combo)
        colormap_layout.addLayout(colorscheme_layout)

        # Flip colormap checkbox
        flip_checkbox_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        self.flip_colormap_checkbox = QCheckBox("Flip colormap")
        self.flip_colormap_checkbox.setChecked(current_settings['flip_colormap'])
        self.flip_colormap_checkbox.stateChanged.connect(self._update_colormap)
        flip_checkbox_layout.addWidget(self.flip_colormap_checkbox)
        colormap_layout.addLayout(flip_checkbox_layout)

        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)

        depth_conversion_group = QGroupBox("Conversion to depth parameters")        
        depth_conversion_layout = QHBoxLayout()

        first_arrival_layout = QHBoxLayout(alignment=Qt.AlignCenter)
        first_arrival_layout.addWidget(QLabel("First arrival: [samples]"))
        self.ind_sample_time_first_arrival_spinbox = QSpinBox()
        self.ind_sample_time_first_arrival_spinbox.setRange(0, 10000)
        self.ind_sample_time_first_arrival_spinbox.setValue(current_settings['ind_sample_time_first_arrival'])
        self.ind_sample_time_first_arrival_spinbox.valueChanged.connect(self._on_image_axes_settings_changed)
        first_arrival_layout.addWidget(self.ind_sample_time_first_arrival_spinbox)
        depth_conversion_layout.addLayout(first_arrival_layout)

        self.velocity_spinboxes = {}
        for key, label in [\
            ("air_velocity_m_per_s", "Air velocity: [m/ns]"),\
            ("ground_velocity_m_per_s", "Ground velocity: [m/ns]")]:
            spinbox = QDoubleSpinBox()
            # convert from meters per second to meter per nanosecond for display
            spinbox.setRange(0.001, C_VACUUM*1e-9)
            spinbox.setValue(current_settings[key]*1e-9)
            spinbox.setSingleStep(0.001)
            spinbox.setDecimals(3)
            spinbox.valueChanged.connect(self._on_image_axes_settings_changed)
            velocity_layout = QHBoxLayout(alignment=Qt.AlignCenter)
            velocity_layout.addWidget(QLabel(label))
            velocity_layout.addWidget(spinbox)
            depth_conversion_layout.addLayout(velocity_layout)
            self.velocity_spinboxes[key] = spinbox

        depth_conversion_group.setLayout(depth_conversion_layout)
        layout.addWidget(depth_conversion_group)


        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


    def _show_hide_colorbar(self):
        self._old_settings['colorbar_visible'] = self.colorbar_visible_checkbox.isChecked()
        self._parent._set_colorbar_visibility(self.colorbar_visible_checkbox.isChecked())


    def _update_colormap(self):
        self._old_settings['colormap'] = self.colormap_combo.currentText()
        self._old_settings['flip_colormap'] = self.flip_colormap_checkbox.isChecked()
        self._parent._update_colormap(scheme = self.colormap_combo.currentText(), flip = self.flip_colormap_checkbox.isChecked())

    def _on_image_axes_settings_changed(self):
        """Called when any setting changes - update parent display immediately."""
        new_settings = self.get_settings_from_layout()
        # Update parent's settings
        self._parent._display_settings = new_settings
        self._parent.redraw_image_ax()
        # Update old settings for next comparison
        self._old_settings = dict(new_settings)

    def reject(self):
        """Restore old settings when dialog is cancelled."""
        self._parent._display_settings = self._old_settings
        self._parent.canvas_render()
        super().reject()

    def get_settings_from_layout(self):
        """Return the selected settings as a dictionary."""
        velocity_settings = {}
        for key, spinbox in self.velocity_spinboxes.items():
            val = spinbox.value()*1e9 # convert from meter per nanosecond to meters per second
            velocity_settings[key] = max(0.001*C_VACUUM, min(val, C_VACUUM))
        ret = {
            'file_name_in_plot': self.file_name_in_plot_checkbox.isChecked(),
            'top': AxisType(self.top_combo.currentText()),
            'top_major_tick': self.top_major_tick.value(),
            'top_minor_ticks': self.top_minor_ticks.value(),
            'bottom': AxisType(self.bottom_combo.currentText()),
            'bottom_major_tick': self.bottom_major_tick.value(),
            'bottom_minor_ticks': self.bottom_minor_ticks.value(),
            'left': AxisType(self.left_combo.currentText()),
            'left_major_tick': self.left_major_tick.value(),
            'left_minor_ticks': self.left_minor_ticks.value(),
            'right': AxisType(self.right_combo.currentText()),
            'right_major_tick': self.right_major_tick.value(),
            'right_minor_ticks': self.right_minor_ticks.value(),
            'colorbar_visible': self.colorbar_visible_checkbox.isChecked(),
            'colormap': self.colormap_combo.currentText(),
            'flip_colormap': self.flip_colormap_checkbox.isChecked(),
            'ind_sample_time_first_arrival': self.ind_sample_time_first_arrival_spinbox.value(),
        }
        ret.update(velocity_settings)
        assert sorted(list(ret.keys())) == sorted(list(DisplaySettingsDialog.default_settings.keys()))
        return ret


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
        self._colorbar_ax: plt.Axes | None = None
        self._image_ax: plt.Axes | None = None
        self._colorbar_indicator: Line2D | None = None
        self._colorbar_indicator_amplitude: float | None = None
        self._colorbar_background = None  # Cache for blitting

        # trace distances data in meters
        self._trace_coords_meters: np.ndarray | None = None
        self._trace_cumulative_distances_meters: np.ndarray | None = None

        # time samples data in seconds
        self._time_interval_seconds: float = 1.0
        self._time_first_arrival_seconds: float = 0.0
        self._trace_time_delays_seconds: np.ndarray | None = None # time delays in seconds for each trace
        self._time_samples_seconds: np.ndarray | None = None
        self._time_display_units: str = "s"
        self._time_display_value_factor: float = 1.0 # for display time in nano seconds, milliseconds, seconds
        self._offset_meters: float = 0.0
        self._depth_converted: np.ndarray | None = None

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

        # Connect draw event to update background cache for blitting
        self._canvas.mpl_connect('draw_event', self._on_draw_complete)

        # Register as listener for global settings changes
        GlobalSettings.add_listener(self._on_global_settings_changed)


    def _get_unit_label(self, axis_type: AxisType) -> str:
        if axis_type == AxisType.TIME:
            return self._time_display_units
        if axis_type in [AxisType.DEPTH, AxisType.DISTANCE]:
            return GlobalSettings.display_length_unit
        return ""

    def _get_axis_geometry_4_display(self, axis_type: AxisType) -> tuple:
        nz, nx = self._data.shape
        if axis_type == AxisType.TIME:
            dt = self._time_interval_seconds*self._time_display_value_factor
            time_min = self._time_samples_seconds[0]*self._time_display_value_factor
            return (time_min, dt, nz)
        if axis_type == AxisType.DISTANCE: # distance between traces
            dx = self._trace_cumulative_distances_meters[-1] / (nx - 1) * GlobalSettings.display_length_factor
            distance_min = 0.0
            return (distance_min, dx, nx)
        if axis_type == AxisType.DEPTH: # time converted to depth
            z_min = self._depth_converted[0] * GlobalSettings.display_length_factor
            z_max = self._depth_converted[-1] * GlobalSettings.display_length_factor
            dz = (z_max - z_min) / (nz - 1)
            return (z_min, dz, nz)
        if axis_type == AxisType.SAMPLE:
            sample_min = 0.0
            return (sample_min, 1, nz)
        if axis_type == AxisType.TRACE:
            trace_min = 0.0
            return (trace_min, 1, nx)
        return (0, 1, 0)

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
        self._image_ax.clear()
        self._image_ax.set_xlim(0, 1)
        self._image_ax.set_ylim(0, 1)
        self._image_ax.axis('off')  # Hide axes

        # Add error icon/symbol at top
        self._image_ax.text(0.5, 0.75, '⚠️',
                       ha='center', va='center',
                       fontsize=80, color='red',
                       transform=self._image_ax.transAxes)

        # Add error title
        self._image_ax.text(0.5, 0.55, title,
                       ha='center', va='center',
                       fontsize=20, weight='bold', color='darkred',
                       transform=self._image_ax.transAxes)

        # Add error message (wrap text if needed)
        self._image_ax.text(0.5, 0.35, message,
                       ha='center', va='center',
                       fontsize=14, color='black',
                       wrap=True,
                       transform=self._image_ax.transAxes)

        # Set background color to light red
        self._image_ax.set_facecolor('#ffebee')

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
        if event.inaxes != self._image_ax:
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
        self._rect = self._image_ax.add_patch(
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
        self._image_ax.set_xlim(max(x_mid - dx, -0.5), min(x_mid + dx, self._data.shape[1] - 0.5))
        self._image_ax.set_ylim(min(y_mid + dy, self._data.shape[0] - 0.5), max(y_mid - dy, -0.5))  # Inverted for image coordinates
        self._canvas.draw_idle()

        self._press_event = None


    def _zoom_out(self) -> None:
        """Zoom out to show full data extent."""
        if self._data is not None:
            self._image_ax.set_xlim(-0.5, self._data.shape[1] - 0.5)
            self._image_ax.set_ylim(self._data.shape[0] - 0.5, -0.5)  # Inverted for image coordinates
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
    def image_ax(self) -> plt.Axes | None:
        """Get the matplotlib image axes."""
        return self._image_ax


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
        """Update the horizontal indicator line on the colorbar using fast blitting."""
        if self._image is None or self._colorbar is None or amplitude is None:
            # Clear indicator if no amplitude
            if self._colorbar_indicator is not None:
                self._remove_colorbar_indicator()
                if self._colorbar_background is not None:
                    self._canvas.restore_region(self._colorbar_background)
                    self._canvas.blit(self._colorbar.ax.bbox)
            return

        # Use blitting for fast updates
        if self._colorbar_background is not None:
            # Restore clean background
            self._canvas.restore_region(self._colorbar_background)

            # Draw new indicator line
            self._create_colorbar_indicator(amplitude)

            # Draw just the indicator
            self._colorbar.ax.draw_artist(self._colorbar_indicator)

            # Blit only the colorbar area (fast!)
            self._canvas.blit(self._colorbar.ax.bbox)
        else:
            # Fallback: no background cached, use full redraw
            self._remove_colorbar_indicator()
            self._create_colorbar_indicator(amplitude)
            self._canvas.draw_idle()


    def get_hover_info(self, x: float|None, z: float|None) -> dict[str, Any] | None:
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

        if x is None:
            ix = -1
        else:
            ix = max(0, min(int(x+0.5), self._data.shape[1] - 1))
        if z is None:
            iz = -1
        else:
            iz = max(0, min(int(z+0.5), self._data.shape[0] - 1))

        amplitude = self._data[iz, ix] if ix >= 0 and iz >= 0 else None

        time_value = simple_interpolation(self._time_samples_seconds, z)
        distance = simple_interpolation(self._trace_cumulative_distances_meters, x)
        depth_value = simple_interpolation(self._depth_converted, z)

        hover_info = {}

        if time_value is not None:
            hover_info['time_units'] = self._time_display_units
            hover_info['time_value'] = time_value * self._time_display_value_factor
        if depth_value is not None:
            hover_info['depth_value'] = depth_value * GlobalSettings.display_length_factor
        if distance is not None:
            hover_info['distance'] = distance * GlobalSettings.display_length_factor
        if amplitude is not None:
            hover_info['amplitude'] = amplitude
        if iz >= 0:
            hover_info['sample_number'] = iz
        if ix >= 0:
            hover_info['trace_number'] = ix
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
            self._remove_colorbar_indicator()
            self._remove_colorbar()
            self._trace_coords_meters = None
            self._trace_cumulative_distances_meters = None
            self._trace_time_delays_seconds = None
            self._time_interval_seconds = 1.0
            self._time_first_arrival_seconds = 0.0
            self._time_samples_seconds = None
            self._depth_converted = None
            self._amplitude_min = 0.0
            self._amplitude_max = 0.0
            self._image = None
            self._filename = ""
            return False

        self._filename = filename
        self._amplitude_min = float(np.min(self._data))
        self._amplitude_max = float(np.max(self._data))
        # Calculate time range in seconds
        nt, nx = self._data.shape
        self._time_first_arrival_seconds = np.median(self._trace_time_delays_seconds)
        time_range_seconds = (nt - 1) * self._time_interval_seconds
        if self._time_first_arrival_seconds < self._time_interval_seconds or self._time_first_arrival_seconds >= time_range_seconds - self._time_interval_seconds:
            # default to 10% of the data range
            self._time_first_arrival_seconds = time_range_seconds * 0.1
        self._time_samples_seconds = np.arange(nt) * self._time_interval_seconds - self._time_first_arrival_seconds
        ind_sample_time_first_arrival = int(self._time_first_arrival_seconds / self._time_interval_seconds)
        while ind_sample_time_first_arrival < self._time_samples_seconds.shape[0] and self._time_samples_seconds[ind_sample_time_first_arrival] < 0:
            ind_sample_time_first_arrival += 1
        self._display_settings['ind_sample_time_first_arrival'] = ind_sample_time_first_arrival
        if time_range_seconds < 0.01:
            # display time in nano seconds
            self._time_display_units = "ns"
            self._time_display_value_factor = 1_000_000_000.0
        elif time_range_seconds < 10.0:
            # display time in milliseconds
            self._time_display_units = "ms"
            self._time_display_value_factor = 1000.0
        else:
            # display time in seconds
            self._time_display_units = "s"
            self._time_display_value_factor = 1.0

        # Calculate cumulative distances along the survey line
        # Distance from trace i-1 to trace i for each trace
        trace_distances_meters = np.sqrt(np.sum(np.diff(self._trace_coords_meters, axis=0)**2, axis=1))
        # Cumulative sum with 0.0 prepended for first trace
        self._trace_cumulative_distances_meters = np.cumsum(np.concatenate([[0.0], trace_distances_meters]))
        assert len(self._trace_cumulative_distances_meters) == len(self._trace_coords_meters)

        self.calculate_depth_converted()

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
                dt_us = float(f.bin[segyio.BinField.Interval])
                if dt_us <= 0:
                    return False, "Sample interval is not set in SEGY file"
                self._time_interval_seconds = dt_us / 1_000_000.0  # Convert microseconds to seconds


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
                factor_length_2_mks = UnitSystem.convert_length_factor(file_unit_system, UnitSystem.MKS)
                time_delays_seconds = np.full(num_traces, fill_value=np.nan)
                trace_offset_meters = np.full(num_traces, fill_value=np.nan)
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
                            x *= factor_length_2_mks
                            y *= factor_length_2_mks
                            pass
                        pass
                    coords[i] = [x, y]
                    trace_offset_meters[i] = trace_header[segyio.TraceField.Offset] * factor_length_2_mks

                if not count_valid_coords:
                    return False, "No trace coordinates found in SEGY file"
                interpolate_inplace_nan_values(coords)
                interpolate_inplace_nan_values(time_delays_seconds)
                interpolate_inplace_nan_values(trace_offset_meters)
                self._trace_time_delays_seconds = time_delays_seconds
                self._offset_meters = np.median(trace_offset_meters)
                self._trace_coords_meters = coords
        except Exception as e:
            return False, f"Error loading SEGY file: {e}"

        return True, ""


    def _load_mala_data(self, filename: str) -> tuple[bool, str]:
        try:
            file_base, _ = os.path.splitext(filename)
            data, info = readMALA(file_base)
            self._data = np.array(data)
            nt, nx = self._data.shape[0], self._data.shape[1]

            # Extract metadata from MALA header
            # Calculate sample interval (dt) from TIMEWINDOW and SAMPLES
            # TIMEWINDOW is in nanoseconds
            timewindow_ns = float(info.get('TIMEWINDOW', 0))
            if timewindow_ns > 0 and nt > 1:
                dt_ns = timewindow_ns / (nt - 1)
                self._time_interval_seconds = dt_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            else:
                self._time_interval_seconds = 1.0

            # Extract signal position (time zero offset) from MALA header
            # SIGNAL POSITION is in nanoseconds
            signal_position_ns = float(info.get('SIGNAL POSITION', 0))
            self._trace_time_delays_seconds = np.full(nx, signal_position_ns / 1_000_000_000.0)  # Convert nanoseconds to seconds

            # Distance interval from MALA header
            distance_interval = float(info.get('DISTANCE INTERVAL', 0))
            if distance_interval <= 0:
                return False, "Distance interval is not set in MALA file"
            # Create linear coordinates based on distance interval
            x_coords = np.arange(nx) * distance_interval
            self._trace_coords_meters = np.column_stack([x_coords, np.zeros(nx)])
            self._offset_meters = info.get('OFFSET', 0)
        except Exception as e:
            return False, f"Error loading MALA file: {e}"

        return True, ""


    def _on_draw_complete(self, event) -> None:
        """Cache colorbar background after draw completes for fast blitting."""
        if self._colorbar is not None and self._data is not None:
            try:
                self._colorbar_background = self._canvas.copy_from_bbox(self._colorbar.ax.bbox)
            except Exception:
                # If copy fails, just skip caching
                self._colorbar_background = None


    def _on_resize(self, event) -> None:
        """Handle canvas resize event to maintain fixed pixel margins."""
        if self._data is not None:
            # Invalidate cached background since layout changes
            self._colorbar_background = None
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

        if self._check_colorbar_ax_visibility():
            colorbar_width_px = GlobalSettings.margins_px['colorbar_width']
            colorbar_right_margin_px = vertical_axes_margin_px + base_horizontal_margin_px
            colorbar_left_margin_px = colorbar_width_px + colorbar_right_margin_px
            colorbar_left = 1 - colorbar_left_margin_px / width_px
            colorbar_width = colorbar_width_px / width_px
            self._colorbar_ax.set_position([colorbar_left, bottom, colorbar_width, height])
            right_image_margin_px += colorbar_left_margin_px

        image_left = left_image_margin_px / width_px
        image_width = 1.0 - (left_image_margin_px + right_image_margin_px) / width_px
        self._image_ax.set_position([image_left, bottom, image_width, height])
        self._canvas.draw_idle()


    def _on_global_settings_changed(self) -> None:
        """Callback when global settings change - update the layout."""
        if self._data is None:
            return
        self._adjust_layout_with_fixed_margins()


    def _recreate_subplots(self) -> None:
        """Recreate subplots with updated parameters."""

        # Clear existing subplots
        self._fig.clear()

        self.create_subplots()
        # Re-render if we have data
        if self._data is not None:
            self.canvas_render()


    def canvas_render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._data is None:
            return

        # Save current zoom state before clearing
        xlim = self._image_ax.get_xlim()
        ylim = self._image_ax.get_ylim()

        self._remove_colorbar()
        self._remove_colorbar_indicator()

        self._image_ax.clear()

        self._apply_display_settings()

        # IMPORTANT: Restore zoom or set initial limits AFTER _apply_display_settings()
        # The _apply_display_settings() method calls imshow(), which triggers matplotlib's
        # autoscaling and can override any axis limits that were set before.
        # Setting limits here (after imshow) ensures they are preserved and prevents the
        # image from being shifted or having white strips at the edges.
        if xlim == (0.0, 1.0):  # Default uninitialized state
            self._image_ax.set_xlim(-0.5, self._data.shape[1] - 0.5)
            self._image_ax.set_ylim(self._data.shape[0] - 0.5, -0.5)
        else:
            self._image_ax.set_xlim(xlim)
            self._image_ax.set_ylim(ylim)

        self._adjust_layout_with_fixed_margins()

        # Force canvas draw to initialize colorbar axis transforms
        self._canvas.draw()

        # Save background for fast blitting of colorbar indicator
        if self._colorbar is not None:
            self._colorbar_background = self._canvas.copy_from_bbox(self._colorbar.ax.bbox)

        self._canvas.draw_idle()


    def create_subplots(self) -> None:
        # Note: We'll set margins and wspace later via _adjust_layout_with_fixed_margins
        self._remove_colorbar_ax()
        if self._display_settings.get('colorbar_visible', True):
            self._image_ax = self._fig.add_subplot(1, 2, 1)
            self._colorbar_ax = self._fig.add_subplot(1, 2, 2)
        else:
            self._image_ax = self._fig.add_subplot(1, 1, 1)
            self._colorbar_ax = None




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


    def _update_colormap(self, scheme: str|None, flip: bool|None):
        if scheme:
            self._display_settings['colormap'] = scheme
        if flip is not None: # work for both values of flip (True/False)
            self._display_settings['flip_colormap'] = flip
        self._remove_colorbar_ax()  # force redraw of the colorbar
        self._check_colorbar_ax_visibility()
        self._adjust_layout_with_fixed_margins()


    def _set_colorbar_visibility(self, visible: bool|int):
        if isinstance(visible, int):
            visible = visible != 0
        self._display_settings['colorbar_visible'] = visible
        self._check_colorbar_ax_visibility()
        self._adjust_layout_with_fixed_margins()


    def redraw_image_ax(self) -> None:
        """Redraw the image_ax of the image."""
        if self._data is None:
            return
        self._apply_display_settings()
        

    def _apply_display_settings(self) -> None:
        """Apply the display settings to the image_ax."""
        if self._data is None:
            return
        # recalculate time samples seconds
        ind_sample_time_first_arrival = self._display_settings['ind_sample_time_first_arrival']
        self._time_samples_seconds = np.arange(-ind_sample_time_first_arrival,self._data.shape[0]-ind_sample_time_first_arrival) * self._time_interval_seconds
        self.calculate_depth_converted()
        colormap = self._display_settings['colormap']
        flip_colormap = self._display_settings['flip_colormap']
        file_name_in_plot = self._display_settings['file_name_in_plot']
        # Add '_r' suffix to flip the colormap
        if flip_colormap:
            colormap = colormap + '_r'
        self._image = self._image_ax.imshow(self._data, aspect="auto", cmap=colormap, vmin=self._amplitude_min, vmax=self._amplitude_max)
        self._check_colorbar_ax_visibility()

        if file_name_in_plot:
            self._image_ax.set_title(os.path.basename(self._filename))

        # Get current settings
        top = self._display_settings['top']
        bottom = self._display_settings['bottom']
        left = self._display_settings['left']
        right = self._display_settings['right']

        # remove top and bottom labels
        self._image_ax.set_xlabel(None)
        self._image_ax.set_ylabel(None)
        self._image_ax.secondary_xaxis('top').set_visible(False)

        # Apply top axis
        if top == AxisType.NONE:
            self._image_ax.xaxis.set_tick_params(top=False, labeltop=False)
            self._image_ax.secondary_xaxis('top').set_visible(False)
        else:
            ax2 = self._image_ax.secondary_xaxis('top')
            ax2.set_visible(True)
            ax2.tick_params(axis='x', top=True, labeltop=True)
            self._apply_tick_settings(ax2.xaxis, top, self._display_settings['top_major_tick'], self._display_settings['top_minor_ticks'])

        # Apply bottom axis
        if bottom == AxisType.NONE:
            self._image_ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
        else:
            self._image_ax.xaxis.set_tick_params(bottom=True, labelbottom=True)
            self._apply_tick_settings(self._image_ax.xaxis, bottom, self._display_settings['bottom_major_tick'], self._display_settings['bottom_minor_ticks'])

        # Apply left axis
        if left == AxisType.NONE:
            self._image_ax.yaxis.set_tick_params(left=False, labelleft=False)
        else:
            self._image_ax.yaxis.set_tick_params(left=True, labelleft=True)
            self._apply_tick_settings(self._image_ax.yaxis, left, self._display_settings['left_major_tick'], self._display_settings['left_minor_ticks'])

        # Apply right axis
        if right == AxisType.NONE:
            self._image_ax.yaxis.set_tick_params(right=False, labelright=False)
            self._image_ax.secondary_yaxis('right').set_visible(False)
        else:
            ax2 = self._image_ax.secondary_yaxis('right')
            ax2.set_visible(True)
            ax2.tick_params(axis='y', right=True, labelright=True)
            self._apply_tick_settings(ax2.yaxis, right, self._display_settings['right_major_tick'], self._display_settings['right_minor_ticks'])


    def _check_colorbar_ax_visibility(self) -> bool:
        colorbar_visible = self._display_settings['colorbar_visible'] # must check with default value True
        if not colorbar_visible or self._data is None or np.isnan(self._amplitude_min) or np.isnan(self._amplitude_max):
            self._remove_colorbar_ax()
            return False
            
        if self._colorbar_ax is not None and self._colorbar_ax.figure is not self._fig:
            self._remove_colorbar_ax()
        if self._colorbar_ax is None:
            if len(self._fig.axes) == 1: # if there is only one subplot, add a new subplot for the colorbar
                self._colorbar_ax = self._fig.add_subplot(1, 2, 2)
            else:
                self._colorbar_ax = self._fig.axes[1]
        self._create_colorbar()
        return True


    def _remove_colorbar_ax(self) -> None:
        self._remove_colorbar()
        #check if colorbar_ax is defined, if so, remove it
        if not hasattr(self, '_colorbar_ax'):
            self._colorbar_ax = None
            return
        if self._colorbar_ax is None:
            return
        if self._colorbar_ax.figure is self._fig and self._colorbar_ax in self._fig.axes:
            self._colorbar_ax.remove()
        self._colorbar_ax = None


    def _remove_colorbar(self) -> None:
        self._remove_colorbar_indicator()
        if self._colorbar is None:
            return

        # Only remove if the colorbar's axes is still in the figure
        if self._colorbar.ax is not None and self._colorbar.ax.figure is self._fig and self._colorbar.ax in self._fig.axes:
            try:
                self._colorbar.remove()
            except (AttributeError, ValueError):
                # Colorbar might already be in an inconsistent state, just clear the reference
                pass

        self._colorbar = None
        self._colorbar_background = None  # Clear cached background


    def _create_colorbar(self):
        if self._colorbar is not None:
            return
        self._remove_colorbar_indicator()
        self._colorbar = self._fig.colorbar(self._image, cax=self._colorbar_ax, label="Amplitude [mV]")
        colorbar_ticks = list(self._colorbar.get_ticks())
        # assuming the colorbar ticks are already sorted, add min and max if not already present
        delta_amplitude = (self._amplitude_max - self._amplitude_min) * 0.05
        # remove ticks that are too close to the min and max
        while colorbar_ticks and colorbar_ticks[0] <= self._amplitude_min + delta_amplitude:
            colorbar_ticks = colorbar_ticks[1:]
        while colorbar_ticks and colorbar_ticks[-1] >= self._amplitude_max - delta_amplitude:
            colorbar_ticks = colorbar_ticks[:-1]
        colorbar_ticks = [self._amplitude_min] + colorbar_ticks + [self._amplitude_max]
        self._colorbar.set_ticks(colorbar_ticks)
        self._create_colorbar_indicator(self._colorbar_indicator_amplitude)


    def _create_colorbar_indicator(self, amplitude: float|None) -> None:
        if self._colorbar_indicator is not None or self._colorbar is None or amplitude is None or self._image is None:
            return

        if abs(self._amplitude_min - self._amplitude_max) < 1e-9:
            norm_value = 0.5
        else:
            norm_value = (amplitude - self._amplitude_min) / (self._amplitude_max - self._amplitude_min)
            norm_value = max(0.0, min(1.0, norm_value))  # Clamp to [0, 1]
        cmap = self._image.get_cmap()
        rgba = cmap(norm_value)
        inv_color = (1.0 - rgba[0], 1.0 - rgba[1], 1.0 - rgba[2])

        self._colorbar_indicator = self._colorbar.ax.axhline(y=amplitude, color=inv_color, linewidth=2, alpha=1.0)
        self._colorbar_indicator_amplitude = amplitude


    def _remove_colorbar_indicator(self) -> None:
        if self._colorbar_indicator is None:
            return        
        self._colorbar_indicator.set_visible(False)
        self._colorbar_indicator = None


    def _apply_tick_settings(self, axis: plt.Axes, axis_type: AxisType, major_tick_distance: float, minor_ticks_per_major: int) -> None:
        """
        Apply tick settings to an axis with a reference offset.

        Args:
            axis: matplotlib axis object (e.g., self._image_ax.xaxis or self._image_ax.yaxis)
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

        if axis_type == AxisType.DEPTH:
            if self._depth_converted is None:
                return
            axis_vector_values = self._depth_converted * GlobalSettings.display_length_factor
        elif axis_type == AxisType.DISTANCE:
            if self._trace_cumulative_distances_meters is None:
                return
            axis_vector_values = self._trace_cumulative_distances_meters * GlobalSettings.display_length_factor
        else:
            axis_vector_values = None
        if axis_vector_values is not None:
            axis_vector_indices = np.arange(len(axis_vector_values))
        else:
            axis_vector_indices = None
        axis_min, axis_step, axis_num_samples = self._get_axis_geometry_4_display(axis_type)

        minor_ticks_per_major = min(minor_ticks_per_major, axis_num_samples)
        axis_max = axis_min + axis_step * (axis_num_samples - 1)
        max_major_tick_distance = max(axis_step, axis_step*(axis_num_samples-1))
        min_major_tick_distance = max(axis_step, 0.001*max_major_tick_distance)
        if major_tick_distance <= min_major_tick_distance:
            major_tick_distance = min_major_tick_distance
            minor_ticks_per_major = 0
        if major_tick_distance >= max_major_tick_distance:
            major_tick_distance = max_major_tick_distance
            minor_ticks_per_major = max(min(10,axis_num_samples), minor_ticks_per_major)
        # Calculate tick positions in display units
        sign_axis_min = int(np.sign(axis_min))
        sign_axis_max = int(np.sign(axis_max))
        n_min = int(abs(axis_min) / major_tick_distance) * sign_axis_min
        n_max = int(abs(axis_max) / major_tick_distance) * sign_axis_max
        if n_min >= n_max:
            # Use automatic tick placement with custom formatter for distance axis
            axis.set_major_locator(plt.AutoLocator())
            axis.set_minor_locator(AutoMinorLocator())
            return

        major_tick_values = np.arange(n_min, n_max + 1) * major_tick_distance
        if axis_vector_values is not None:
            major_tick_positions = np.interp(major_tick_values, axis_vector_values, axis_vector_indices)
        else:
            major_tick_positions = (major_tick_values - axis_min) / axis_step
        # Set ticks at data coordinate positions with display unit labels
        axis.set_ticks(major_tick_positions)
        axis.set_ticklabels([format_value(val, 3) for val in major_tick_values])

        if minor_ticks_per_major < 2:
            axis.set_minor_locator(plt.NullLocator())
            return

        # Calculate minor tick positions
        minor_tick_distance = major_tick_distance / minor_ticks_per_major
        n_min_minor = int(abs(axis_min) / minor_tick_distance) * sign_axis_min
        n_max_minor = int(abs(axis_max) / minor_tick_distance) * sign_axis_max
        minor_tick_values = np.arange(n_min_minor, n_max_minor + 1)
        minor_tick_values = minor_tick_values[minor_tick_values % minor_ticks_per_major != 0] * minor_tick_distance

        # Convert to data coordinates
        if axis_vector_values is not None:
            minor_tick_positions = np.interp(minor_tick_values, axis_vector_values, axis_vector_indices)
        else:
            minor_tick_positions = (minor_tick_values - axis_min) / axis_step
        axis.set_ticks(minor_tick_positions, minor=True)


    def calculate_depth_converted(self) -> None:
        """
        Calculate depth for bistatic GPR using geometric correction.

        For GPR with fixed Tx-Rx antenna separation (offset), the signal travels
        diagonally from transmitter to reflector and back to receiver. The geometry
        forms a triangle where:
        - Slant distance: L = sqrt(d² + (offset/2)²)
        - Two-way time: t = 2L/v
        - Solving for depth: d = sqrt((v*t/2)² - (offset/2)²)
        """
        if self._time_samples_seconds is None:
            return
        air_velocity_m_per_s = min(self._display_settings['air_velocity_m_per_s'], C_VACUUM)
        ground_velocity_m_per_s = min(self._display_settings['ground_velocity_m_per_s'], C_VACUUM)
        n_time_samples = len(self._time_samples_seconds)
        critical_time = self._offset_meters / ground_velocity_m_per_s
        half_offset = 0.5 * self._offset_meters
        ind_sample_time_first_arrival = self._display_settings['ind_sample_time_first_arrival']
        self._depth_converted = np.empty_like(self._time_samples_seconds)
        ind_sample_critical_time = ind_sample_time_first_arrival
        while ind_sample_critical_time < n_time_samples and self._time_samples_seconds[ind_sample_critical_time] < critical_time:
            ind_sample_critical_time += 1
        # Above surface (negative time): antenna height above ground
        # Uses air velocity for propagation before first arrival
        self._depth_converted[:ind_sample_time_first_arrival] = \
            self._time_samples_seconds[:ind_sample_time_first_arrival] * air_velocity_m_per_s

        # Below surface: geometric correction for bistatic antenna configuration
        # Signal travels diagonally from Tx to reflector to Rx
        extra_time = (ind_sample_critical_time - ind_sample_time_first_arrival) * self._time_interval_seconds
        two_way_time = self._time_samples_seconds[ind_sample_time_first_arrival:] + extra_time
        slant_distance = (two_way_time * ground_velocity_m_per_s) / 2

        # For near-surface where slant_distance < half_offset (geometrically impossible),
        # clip to zero depth. This handles the "direct wave zone" near the surface.
        self._depth_converted[ind_sample_time_first_arrival:] = np.sqrt(np.maximum(slant_distance**2 - half_offset**2, 0))


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


