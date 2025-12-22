from matplotlib.transforms import Bbox
import numpy as np
import os
from typing import Any
from enum import Enum
import cv2

from PySide6.QtWidgets import (
    QMessageBox,
    QVBoxLayout,
    QFileDialog,
    QMenu,
    QFrame,
    QDialog,
    QSizePolicy,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QGridLayout,
)

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.image as plt_image

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
from filters import FilterPipeline, FiltersDialog
from data_file import DataFile

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

vertical_axis_types: list[AxisType] = [AxisType.DEPTH, AxisType.SAMPLE, AxisType.TIME]
horizontal_axis_types: list[AxisType] = [AxisType.DISTANCE, AxisType.TRACE]

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
        self.file_name_in_plot_checkbox.stateChanged.connect(self._on_file_name_in_plot_settings_changed)
        layout.addWidget(self.file_name_in_plot_checkbox)

        # Create grid layout for axes settings (table format without titles)
        image_axes_group = QGroupBox("Image's axes properties")
        grid_layout = QGridLayout()

        # Row 0: Top axis
        grid_layout.addWidget(QLabel("Top:"), 0, 0)
        self.top_combo = QComboBox()
        self.top_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.top_combo.setCurrentText(current_settings['top'].value)
        self.top_combo.currentIndexChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.top_combo, 0, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 0, 2)
        self.top_major_tick = QSpinBox()
        self.top_major_tick.setRange(0, 10000)
        self.top_major_tick.setValue(current_settings['top_major_tick'])
        self.top_major_tick.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.top_major_tick, 0, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 0, 4)
        self.top_minor_ticks = QSpinBox()
        self.top_minor_ticks.setRange(0, 100)
        self.top_minor_ticks.setValue(current_settings['top_minor_ticks'])
        self.top_minor_ticks.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.top_minor_ticks, 0, 5)

        # Row 1: Bottom axis
        grid_layout.addWidget(QLabel("Bottom:"), 1, 0)
        self.bottom_combo = QComboBox()
        self.bottom_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.horizontal_options))
        self.bottom_combo.setCurrentText(current_settings['bottom'].value)
        self.bottom_combo.currentIndexChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.bottom_combo, 1, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 1, 2)
        self.bottom_major_tick = QSpinBox()
        self.bottom_major_tick.setRange(0, 10000)
        self.bottom_major_tick.setValue(current_settings['bottom_major_tick'])
        self.bottom_major_tick.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.bottom_major_tick, 1, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 1, 4)
        self.bottom_minor_ticks = QSpinBox()
        self.bottom_minor_ticks.setRange(0, 100)
        self.bottom_minor_ticks.setValue(current_settings['bottom_minor_ticks'])
        self.bottom_minor_ticks.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.bottom_minor_ticks, 1, 5)

        # Row 2: Left axis
        grid_layout.addWidget(QLabel("Left:"), 2, 0)
        self.left_combo = QComboBox()
        self.left_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.left_combo.setCurrentText(current_settings['left'].value)
        self.left_combo.currentIndexChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.left_combo, 2, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 2, 2)
        self.left_major_tick = QSpinBox()
        self.left_major_tick.setRange(0, 10000)
        self.left_major_tick.setValue(current_settings['left_major_tick'])
        self.left_major_tick.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.left_major_tick, 2, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 2, 4)
        self.left_minor_ticks = QSpinBox()
        self.left_minor_ticks.setRange(0, 100)
        self.left_minor_ticks.setValue(current_settings['left_minor_ticks'])
        self.left_minor_ticks.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.left_minor_ticks, 2, 5)

        # Row 3: Right axis
        grid_layout.addWidget(QLabel("Right:"), 3, 0)
        self.right_combo = QComboBox()
        self.right_combo.addItems(self._enum_to_string_list(DisplaySettingsDialog.vertical_options))
        self.right_combo.setCurrentText(current_settings['right'].value)
        self.right_combo.currentIndexChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.right_combo, 3, 1)
        grid_layout.addWidget(QLabel("Major tick:"), 3, 2)
        self.right_major_tick = QSpinBox()
        self.right_major_tick.setRange(0, 10000)
        self.right_major_tick.setValue(current_settings['right_major_tick'])
        self.right_major_tick.valueChanged.connect(self._on_image_four_axes_settings_changed)
        grid_layout.addWidget(self.right_major_tick, 3, 3)
        grid_layout.addWidget(QLabel("Minor ticks per major:"), 3, 4)
        self.right_minor_ticks = QSpinBox()
        self.right_minor_ticks.setRange(0, 100)
        self.right_minor_ticks.setValue(current_settings['right_minor_ticks'])
        self.right_minor_ticks.valueChanged.connect(self._on_image_four_axes_settings_changed)
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
        self.ind_sample_time_first_arrival_spinbox.valueChanged.connect(self._on_depth_conversion_settings_changed)
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
            spinbox.valueChanged.connect(self._on_depth_conversion_settings_changed)
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

    def _on_file_name_in_plot_settings_changed(self):
        self._old_settings['file_name_in_plot'] = self.file_name_in_plot_checkbox.isChecked()
        self._parent._update_file_name_in_plot(self.file_name_in_plot_checkbox.isChecked())
        self._parent._canvas.draw_idle()


    def _on_depth_conversion_settings_changed(self):
        for key, spinbox in self.velocity_spinboxes.items():
            self._old_settings[key] = spinbox.value() * 1e9 # convert from meter per nanosecond to meters per second
            self._parent._display_settings[key] = spinbox.value() * 1e9 # convert from meter per nanosecond to meters per second

        gui_requested_first_arrival_sample = self.ind_sample_time_first_arrival_spinbox.value()
        result_first_arrival_sample = self._parent._on_change_first_arrival_sample(gui_requested_first_arrival_sample)
        self._old_settings['ind_sample_time_first_arrival'] = result_first_arrival_sample
        self.ind_sample_time_first_arrival_spinbox.blockSignals(True)
        self.ind_sample_time_first_arrival_spinbox.setValue(result_first_arrival_sample)
        self.ind_sample_time_first_arrival_spinbox.blockSignals(False)


    def _on_image_four_axes_settings_changed(self):
        """Called when any setting changes - update parent display immediately."""
        new_settings = self.get_settings_from_layout()
        # Update old settings for next comparison
        self._old_settings = dict(new_settings)
        # Update parent's settings
        self._parent._display_settings = new_settings
        self._parent._apply_image_four_axes_tick_settings()
        self._parent._canvas.draw_idle()


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

    HOME_MODE_FIT = "fit"
    HOME_MODE_1X1 = "1x1"

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        # Call parent init (will call empty on_create)
        super().__init__(main_window, parent)

        self._data_file: DataFile | None = None
        self._processed_data: np.ndarray | None = None  # Cache for filtered data

        self._filter_pipeline = FilterPipeline()
        self._file_view_region: Bbox | None = None
        self._file_region_clipped: Bbox | None = None
        self._canvas_render_region: Bbox | None = None
        self._canvas_buffer: np.ndarray | None = None
        self._amplitude_min: float = 0.0
        self._amplitude_max: float = 0.0
        self._colorbar: plt.Colorbar | None = None
        self._image: plt_image.AxesImage | None = None
        self._colorbar_ax: plt.Axes | None = None
        self._image_ax: plt.Axes | None = None
        self._colorbar_indicator: Line2D | None = None
        self._colorbar_background = None  # Cache for blitting

        self._trace_cumulative_distances_meters: np.ndarray | None = None
        self._trace_cumulative_distances_display: np.ndarray | None = None

        # time samples data in seconds
        self._time_first_arrival_seconds: float = 0.0
        self._time_axis_values_seconds: np.ndarray | None = None
        self._time_axis_values_display: np.ndarray | None = None
        self._time_display_units: str = "s"
        self._time_display_value_factor: float = 1.0 # for display time in nano seconds, milliseconds, seconds
        self._depth_converted_meters: np.ndarray | None = None
        self._depth_converted_display: np.ndarray | None = None
        self._horizontal_indices_in_data_region: np.ndarray | None = None
        self._vertical_indices_in_data_region: np.ndarray | None = None
        # Home button mode: "fit" = fit to window, "1:1" = 1:1 pixels
        self._home_mode: str = SeismicSubWindow.HOME_MODE_FIT

        # Display settings
        self._display_settings: dict[str, Any] = dict(DisplaySettingsDialog.default_settings)

        self._fig = Figure(figsize=(10, 6))

        self.create_subplots()
        self._canvas = FigureCanvasQTAgg(self._fig)

        # Create matplotlib navigation toolbar (for pan/zoom)
        self._nav_toolbar = NavigationToolbar2QT(self._canvas, self)
        # Hide coordinate display in toolbar
        self._nav_toolbar.set_message = lambda x: None
        # Remove internal margins so separators can span full height
        self._nav_toolbar.layout().setContentsMargins(0, 0, 0, 0)

        # Replace "Configure subplots" action with Display Settings
        self._replace_configure_subplots_action()

        # Replace "Home" action with custom behavior
        self._replace_home_action()

        # Replace "Save" action to show save menu
        self._replace_save_action()

        # Add Open button to toolbar
        self._add_open_button()

        # Add Filters button to toolbar
        self._add_filters_button()

        # Add context menu to NavigationToolbar
        self._nav_toolbar.setContextMenuPolicy(Qt.CustomContextMenu)
        self._nav_toolbar.customContextMenuRequested.connect(self._show_nav_toolbar_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._nav_toolbar)
        layout.addWidget(self._canvas)

        self.setMinimumSize(400, 300)

        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect canvas hover events to propagate to main window
        self._canvas.mpl_connect("axes_leave_event", self._axes_leave_event)
        self._canvas.mpl_connect("motion_notify_event", self._motion_notify_event)

        # Connect scroll event for mouse wheel zoom
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

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


    def _get_axis_values_for_display(self, axis_type: AxisType) -> tuple[np.ndarray, np.ndarray] | None:
        if self._canvas_render_region is None:
            return None
        if axis_type in vertical_axis_types:
            samples = np.arange(self._canvas_render_region.y0, self._canvas_render_region.y1, dtype=int)
        elif axis_type in horizontal_axis_types:
            samples = np.arange(self._canvas_render_region.x0, self._canvas_render_region.x1, dtype=int)
        else:
            return None
        if axis_type == AxisType.DEPTH:
            if self._depth_converted_display is None:
                return None
            return samples, self._depth_converted_display
        if axis_type == AxisType.DISTANCE:
            if self._trace_cumulative_distances_display is None:
                return None
            return samples, self._trace_cumulative_distances_display
        if axis_type == AxisType.TIME:  
            if self._time_axis_values_display is None:
                return None
            return samples, self._time_axis_values_display
        if axis_type == AxisType.SAMPLE:
            if self._vertical_indices_in_data_region is None:
                return None
            return samples, self._vertical_indices_in_data_region
        if axis_type == AxisType.TRACE:
            if self._horizontal_indices_in_data_region is None:
                return None
            return samples, self._horizontal_indices_in_data_region
        return None
    

    def _get_unit_factor_for_display(self, axis_type: AxisType) -> float:
        if axis_type == AxisType.TIME:
            return self._time_display_value_factor
        if axis_type in [AxisType.DEPTH, AxisType.DISTANCE]:
            return GlobalSettings.display_length_factor
        return 1.0


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


    def _show_nav_toolbar_context_menu(self, position) -> None:
        """Show context menu for NavigationToolbar (only for specific buttons)."""
        # Find which action is under the cursor
        action_at_cursor = self._nav_toolbar.actionAt(position)

        if action_at_cursor is None:
            return  # No action under cursor, don't show menu

        # Get the action's text or tooltip to identify it
        action_text = action_at_cursor.text()
        action_tooltip = action_at_cursor.toolTip()

        # Only show context menu for specific toolbar buttons
        allowed_buttons = ["Home", "Back", "Forward", "Pan", "Zoom"]

        # Check if this action is one of the allowed buttons
        is_allowed = False
        for button_name in allowed_buttons:
            if button_name in action_text or button_name in action_tooltip:
                is_allowed = True
                break

        if not is_allowed:
            return  # Not an allowed button, don't show menu

        # Show the context menu
        context_menu = QMenu(self)

        # Home button behavior submenu
        home_menu = QMenu("Home Button Behavior", self)

        # Fit to Window option
        fit_to_window_action = QAction("Fit to Window", self)
        fit_to_window_action.setCheckable(True)
        fit_to_window_action.setChecked(self._home_mode == SeismicSubWindow.HOME_MODE_FIT)
        fit_to_window_action.triggered.connect(lambda: self._set_home_mode(SeismicSubWindow.HOME_MODE_FIT))
        home_menu.addAction(fit_to_window_action)

        # 1:1 Pixels option
        one_to_one_action = QAction("1:1 Pixels", self)
        one_to_one_action.setCheckable(True)
        one_to_one_action.setChecked(self._home_mode == SeismicSubWindow.HOME_MODE_1X1)
        one_to_one_action.triggered.connect(lambda: self._set_home_mode(SeismicSubWindow.HOME_MODE_1X1))
        home_menu.addAction(one_to_one_action)

        context_menu.addMenu(home_menu)

        # Show menu at cursor position
        context_menu.exec(self._nav_toolbar.mapToGlobal(position))


    def _set_home_mode(self, mode: str) -> None:
        """Set the home button behavior mode."""
        if mode not in [SeismicSubWindow.HOME_MODE_FIT, SeismicSubWindow.HOME_MODE_1X1]:
            return
        self._home_mode = mode
        self._home()


    def _home(self) -> None:
        self._reset_view_file_region_to_home()
        self._set_canvas_to_image()


    def _update_image_extent_and_limits(self) -> None:
        """Set image extent, limits, and aspect based on home mode."""
        if self._image is None or self._canvas_buffer is None:
            return
        # Extent covers full buffer (what imshow displays)
        h, w = self._canvas_buffer.shape
        self._image.set_extent([-0.5, w - 0.5, h - 0.5, -0.5])
        # xlim/ylim depend on home mode
        if self._home_mode == SeismicSubWindow.HOME_MODE_1X1:
            # 1:1 mode: show full buffer (data at actual pixel size with padding)
            self._image_ax.set_xlim(-0.5, w - 0.5)
            self._image_ax.set_ylim(h - 0.5, -0.5)  # Inverted
        elif self._canvas_render_region is not None:
            # Fit mode: show only the data region (fills the view)
            x0, y0 = self._canvas_render_region.x0, self._canvas_render_region.y0
            x1, y1 = self._canvas_render_region.x1, self._canvas_render_region.y1
            self._image_ax.set_xlim(x0 - 0.5, x1 - 0.5)
            self._image_ax.set_ylim(y1 - 0.5, y0 - 0.5)  # Inverted
        # Always 'auto' - 1:1 pixel mapping is handled by resampling, not matplotlib aspect
        self._image_ax.set_aspect('auto')

    def _set_canvas_to_image(self) -> None:
        """Set the canvas to the image."""
        if self._image is None or self._canvas_buffer is None:
            return
        assert self._get_pixels_shape() == self._get_canvas_buffer_shape(), f'_set_canvas_to_image() shape mismatch: {self._get_pixels_shape()=} != {self._get_canvas_buffer_shape()=}'
        self._image.set_data(self._canvas_buffer)
        self._update_image_extent_and_limits()
        self._image.set_clim(vmin=self._amplitude_min, vmax=self._amplitude_max)
        self._apply_image_four_axes_tick_settings()
        self._canvas.draw() # force draw


    def _reset_view_file_region_to_home(self) -> None:
        """Set the view file region by mode."""
        if self._home_mode == SeismicSubWindow.HOME_MODE_FIT:
            self._file_view_region_fit_to_window()
        elif self._home_mode == SeismicSubWindow.HOME_MODE_1X1:
            self._file_view_region_one_to_one_pixels()
        self._resample_axis_values_4_display()


    def _file_view_region_fit_to_window(self) -> None:
        """Reset view to show entire image fitted to window."""
        display_data = self._processed_data
        if display_data is None:
            return
        nz, nx = display_data.shape
        canvas_shape = self._get_pixels_shape()
        try:
            self._file_view_region = Bbox([[0, 0], [nx, nz]])
            self._file_region_clipped = Bbox(self._file_view_region)
            self._canvas_render_region = Bbox([[0, 0], [canvas_shape[1], canvas_shape[0]]])
            self._canvas_buffer = np.empty(shape=canvas_shape, dtype=display_data.dtype)
            # without crop. only resize to allocated buffer.
            # dsize is (width, height), opposite of numpy shape (height, width)
            cv2.resize(display_data, dsize=(canvas_shape[1], canvas_shape[0]), interpolation=cv2.INTER_CUBIC, dst=self._canvas_buffer)
        except Exception as e:
            self._show_error("Error", f"Failed to set file view region to fit to window: {e}")
            self._file_view_region = None
            self._file_region_clipped = None
            self._canvas_render_region = None
            self._canvas_buffer = None


    def _file_view_region_one_to_one_pixels(self) -> None:
        """Reset view to 1:1 pixel ratio (actual size), showing only part of image if needed."""
        display_data = self._processed_data
        if display_data is None:
            return

        # Calculate how many data pixels can fit in the canvas at 1:1 ratio
        # At 1:1, one data pixel = one screen pixel
        canvas_shape = self._get_pixels_shape()
        data_shape = display_data.shape
        render_shape = (min(data_shape[0], canvas_shape[0]), min(data_shape[1], canvas_shape[1]))

        self._file_view_region = Bbox([[0, 0], [render_shape[1], render_shape[0]]])
        self._file_region_clipped = Bbox([[0, 0], [render_shape[1], render_shape[0]]])
        self._canvas_render_region = Bbox([[0, 0], [render_shape[1], render_shape[0]]])
        self._canvas_buffer = np.full(shape=canvas_shape, fill_value=np.nan, dtype=np.float32)
        self._canvas_buffer[:render_shape[0], :render_shape[1]] = display_data[:render_shape[0], :render_shape[1]]


    @staticmethod
    def is_shape_within_shape(shape1: tuple[int, int], shape2: tuple[int, int]) -> bool:
        """Check if shape1 is within shape2."""
        return shape1[0] <= shape2[0] and shape1[1] <= shape2[1]


    def _get_pixels_shape(self) -> tuple[int, int]:
        """Get the shape of the image axes in screen pixels."""
        if self._image_ax is None:
            return (0, 0)
        renderer = self._canvas.get_renderer()
        bbox = self._image_ax.get_window_extent(renderer=renderer)
        return (int(bbox.height), int(bbox.width))


    def _get_file_shape(self) -> tuple[int, int]:
        """Get the shape of the processed data in samples."""
        if self._processed_data is None:
            return (0, 0)
        return (int(self._processed_data.shape[0]), int(self._processed_data.shape[1]))


    def _get_file_view_region_shape(self) -> tuple[int, int]:
        """Get the shape of the file view region in pixels."""
        if self._file_view_region is None:
            return (0, 0)
        return (int(self._file_view_region.y1 - self._file_view_region.y0), int(self._file_view_region.x1 - self._file_view_region.x0))


    def _get_file_region_clipped_shape(self) -> tuple[int, int]:
        """Get the shape of the file region clipped in samples."""
        if self._file_region_clipped is None:
            return (0, 0)
        return (int(self._file_region_clipped.y1 - self._file_region_clipped.y0), int(self._file_region_clipped.x1 - self._file_region_clipped.x0))


    def _get_canvas_buffer_shape(self) -> tuple[int, int]:
        """Get the shape of the canvas buffer."""
        if self._canvas_buffer is None:
            return (0, 0)
        return (int(self._canvas_buffer.shape[0]), int(self._canvas_buffer.shape[1]))


    def _get_canvas_render_region_shape(self) -> tuple[int, int]:
        """Get the shape of the canvas render region."""
        if self._canvas_render_region is None:
            return (0, 0)
        return (int(self._canvas_render_region.y1 - self._canvas_render_region.y0), int(self._canvas_render_region.x1 - self._canvas_render_region.x0))
        

    def _on_scroll_zoom(self, event) -> None:
        """Handle mouse wheel scroll for zooming (like Google Maps)."""
        if event.inaxes != self._image_ax:
            return  # Only zoom when over the image axes

        if self._canvas_buffer is None or self._file_view_region is None:
            return

        # Zoom factor: scroll up = zoom in, scroll down = zoom out
        zoom_factor = 1.2
        if event.button == 'down':
             zoom_factor = 1/zoom_factor

        # Get current axis limits
        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata
        cur_yrange, cur_xrange = self._get_pixels_shape()
        # Calculate what fraction of the current range the mouse is at
        # This keeps the point under the cursor fixed
        rel_x = xdata / cur_xrange
        rel_y = ydata / cur_yrange
        # the requested region (in files coordinates) corresponds exactly to the entire canvas area (in pixels coordinates)
        old_xrange_request = self._file_view_region.x1 - self._file_view_region.x0
        old_yrange_request = self._file_view_region.y1 - self._file_view_region.y0
        # relative position in the canvas corresponds to the same relative postion in the file request area
        file_x_data = self._file_view_region.x0 + old_xrange_request * rel_x
        file_y_data = self._file_view_region.y0 + old_yrange_request * rel_y
        # Calculate new range (smaller range = more zoomed in)
        new_xrange_request = old_xrange_request / zoom_factor
        new_yrange_request = old_yrange_request / zoom_factor
        # Calculate new limits centered on the mouse position
        new_x0 = file_x_data - new_xrange_request * rel_x
        new_x1 = file_x_data + new_xrange_request * (1 - rel_x)
        new_y0 = file_y_data - new_yrange_request * rel_y
        new_y1 = file_y_data + new_yrange_request * (1 - rel_y)
        self._set_file_view_region(new_x0, new_y0, new_x1, new_y1)
        self._recreate_canvas_buffer()
        self._set_canvas_to_image()


    def _set_file_view_region(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Set the crop request."""
        if self._processed_data is None:
            return
        file_shape = self._get_file_shape()

        def fix_lim(lim0: float, lim1: float, shape: int) -> tuple[float, float]:
            if lim1 < lim0:
                lim0, lim1 = lim1, lim0
            min_range = 2
            max_range = shape*2
            while True:
                lim_range = abs(int(lim1) - int(lim0))
                if min_range <= lim_range <= max_range:
                    break
                lim_center = (int(lim0) + int(lim1)) / 2
                if lim_range < min_range:
                    lim_range = min_range + 1
                elif lim_range > max_range:
                    lim_range = max_range - 1
                lim0 = int(lim_center - lim_range/2)
                lim1 = int(lim_center + lim_range/2)
            # ensure the limits are at least the minimum range inside the shape of the file
            shift_lim = max(0, min_range - lim1) + max(0, lim0 - (max_range - min_range))
            lim0 = lim0 + shift_lim
            lim1 = lim1 + shift_lim
            return lim0, lim1

        x0, x1 = fix_lim(x0, x1, file_shape[1])
        y0, y1 = fix_lim(y0, y1, file_shape[0])
        if x0 >= x1 or y0 >= y1:
            return

        self._file_view_region = Bbox([[int(x0), int(y0)], [int(x1), int(y1)]])
        clipped_x0 = max(0, int(x0))
        clipped_y0 = max(0, int(y0))
        clipped_x1 = min(file_shape[1], int(x1))
        clipped_y1 = min(file_shape[0], int(y1))
        if self._home_mode == SeismicSubWindow.HOME_MODE_1X1:
            display_shape = self._get_pixels_shape()
            clipped_w = clipped_x1 - clipped_x0
            clipped_h = clipped_y1 - clipped_y0
            clipped_x1 = clipped_x0 + min(clipped_w, display_shape[1])
            clipped_y1 = clipped_y0 + min(clipped_h, display_shape[0])
        self._file_region_clipped = Bbox([[clipped_x0, clipped_y0], [clipped_x1, clipped_y1]])
        self._canvas_render_region = None
        self._canvas_buffer = None


    def _check_canvas_buffer_shape(self) -> None:
        """Check if the canvas render region is valid."""
        pixels_shape = self._get_pixels_shape()
        if pixels_shape[0] < 1 or pixels_shape[1] < 1:
            self._canvas_render_region = None
            self._canvas_buffer = None
            return
        if self._canvas_buffer is not None and self._canvas_buffer.shape == pixels_shape:
            return
        self._recreate_canvas_buffer()


    def _recreate_canvas_buffer(self) -> None:
        """Resize the cropped data to the new size."""
        display_data = self._processed_data
        if display_data is None or self._file_region_clipped is None or self._file_view_region is None:
            self._canvas_render_region = None
            self._canvas_buffer = None
            return
        file_view_region_shape = self._get_file_view_region_shape()
        request_height, request_width = file_view_region_shape
        pixels_shape = self._get_pixels_shape()
        try:
            if self._home_mode == SeismicSubWindow.HOME_MODE_1X1:
                # using clipped file region to create canvas render region
                file_clipped_shape = self._get_file_region_clipped_shape()
                assert SeismicSubWindow.is_shape_within_shape(file_clipped_shape, pixels_shape),\
                    f'_recreate_canvas_buffer() {self._home_mode=} {file_clipped_shape=} < {pixels_shape=}'
                self._canvas_render_region = Bbox([[0, 0], [file_clipped_shape[1], file_clipped_shape[0]]])
            else:
                self._canvas_render_region = Bbox([\
                    [int((self._file_region_clipped.x0 - self._file_view_region.x0)/request_width*pixels_shape[1]),\
                    int((self._file_region_clipped.y0 - self._file_view_region.y0)/request_height*pixels_shape[0])],\
                    [int((self._file_region_clipped.x1 - self._file_view_region.x0)/request_width*pixels_shape[1]),\
                    int((self._file_region_clipped.y1 - self._file_view_region.y0)/request_height*pixels_shape[0])]
                    ])
            self._canvas_buffer = np.full(shape=pixels_shape, fill_value=np.nan, dtype=np.float32)
            # dsize is (width, height), opposite of numpy shape (height, width)
            display_crop = self._canvas_render_region
            file_crop = self._file_region_clipped
            dsize_width = int(display_crop.x1) - int(display_crop.x0)
            dsize_height = int(display_crop.y1) - int(display_crop.y0)
            cv2.resize(\
                src=display_data[int(file_crop.y0):int(file_crop.y1), int(file_crop.x0):int(file_crop.x1)],\
                dsize=(dsize_width, dsize_height),\
                interpolation=cv2.INTER_CUBIC,\
                dst=self._canvas_buffer[int(display_crop.y0):int(display_crop.y1), int(display_crop.x0):int(display_crop.x1)])
            self._resample_axis_values_4_display()
        except Exception as e:
            self._show_error("Error", f"Failed to recreate canvas buffer: {e}")
            self._canvas_render_region = None
            self._canvas_buffer = None


    @staticmethod
    def create_from_load_file(main_window: UASMainWindow) -> None:
        """Create a new seismic subwindow from a loaded file."""
        subwindow = SeismicSubWindow(main_window)
        if subwindow.load_file_dialog():
            main_window.add_subwindow(subwindow)
        else:
            subwindow.deleteLater()


    @property
    def filename(self) -> str:
        """Get the loaded filename."""
        return self._data_file.filename if self._data_file is not None else ""


    @property
    def raw_data(self) -> np.ndarray | None:
        """Get the raw data from the loaded data file."""
        return self._data_file.data if self._data_file is not None else None


    @property
    def time_interval_seconds(self) -> float:
        """Get the time interval in seconds."""
        return self._data_file.time_interval_seconds if self._data_file is not None else 1.0


    @property
    def offset_meters(self) -> float:
        """Get the Tx-Rx offset in meters."""
        return self._data_file.offset_meters if self._data_file is not None else 0.0


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
        if self._colorbar is None or amplitude is None:
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

            # Remove old indicator and draw new one at new position
            self._remove_colorbar_indicator()
            self._create_colorbar_indicator(amplitude)

            # Draw just the indicator
            if self._colorbar_indicator is not None:
                self._colorbar.ax.draw_artist(self._colorbar_indicator)

            # Blit only the colorbar area (fast!)
            self._canvas.blit(self._colorbar.ax.bbox)
        else:
            # Fallback: no background cached, use full redraw
            self._remove_colorbar_indicator()
            self._create_colorbar_indicator(amplitude)
            self._canvas.draw_idle()


    def get_hover_info(self, x: float|None, y: float|None) -> dict[str, Any] | None:
        """
        Calculate hover information for given pixel coordinates.

        Args:
            x: Trace number (horizontal pixel coordinate)
            y: Sample number (vertical pixel coordinate)

        Returns:
            Dictionary with hover info, or None if out of bounds
        """
        data = self._canvas_buffer
        if data is None or self._canvas_render_region is None:
            assert self.raw_data is None,\
                f"get_hover_info: {(self._canvas_buffer is None)=}, {(self._canvas_render_region is None)=}\n" +\
                f"The only time this should happen is when the file is NOT loaded. However: {(self.raw_data is not None)=}" 
            return None

        hover_info = {}

        if x is None:
            ix = -1
        elif x < self._canvas_render_region.x0 - 0.5 or x > self._canvas_render_region.x1 - 0.5:
            ix = -1
        else:
            ix = max(0, min(int(x+0.5), data.shape[1] - 1))
            x_in_data_region = x - self._canvas_render_region.x0  # x is the pixel coordinate in the display
            ind_trace = simple_interpolation(self._horizontal_indices_in_data_region, x_in_data_region)
            hover_info['trace_number'] = ind_trace
            distance = simple_interpolation(self._trace_cumulative_distances_display, x_in_data_region)
            hover_info['distance'] = distance

        if y is None:
            iy = -1
        elif y < self._canvas_render_region.y0 - 0.5 or y > self._canvas_render_region.y1 - 0.5:
            iy = -1
        else:
            iy = max(0, min(int(y+0.5), data.shape[0] - 1))
            y_in_data_region = y - self._canvas_render_region.y0  # y is the pixel coordinate in the display
            ind_sample = simple_interpolation(self._vertical_indices_in_data_region, y_in_data_region)
            hover_info['sample_number'] = ind_sample
            time_value = simple_interpolation(self._time_axis_values_display, y_in_data_region)
            hover_info['time_units'] = self._time_display_units
            hover_info['time_value'] = time_value
            depth_value = simple_interpolation(self._depth_converted_display, y_in_data_region)
            hover_info['depth_value'] = depth_value

        if ix >= 0 and iy >= 0:
            amplitude = data[iy, ix]
            hover_info['amplitude'] = amplitude
            
        return hover_info


    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including loaded file and color scale."""
        state = super().serialize()
        state['filename'] = self.filename

        # Convert AxisType enums to strings for JSON serialization
        serialized_settings = dict(self._display_settings)
        for key in ['top', 'bottom', 'left', 'right']:
            if key in serialized_settings and isinstance(serialized_settings[key], AxisType):
                serialized_settings[key] = serialized_settings[key].value

        state['display_settings'] = serialized_settings

        # Serialize filter pipeline
        state['filter_pipeline'] = self._filter_pipeline.serialize()

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

        # Restore filter pipeline
        if 'filter_pipeline' in state:
            self._filter_pipeline.deserialize(state['filter_pipeline'])
        self.load_file(state.get("filename", ""))
        self._apply_image_four_axes_tick_settings()
        self.canvas_render()


    def _create_save_menu(self) -> QMenu:
        """Create and return the save submenu with all format options."""
        save_menu = QMenu("Save as...", self)

        # Save as PNG (uses matplotlib's save dialog)
        save_png_action = QAction("As PNG Image...", self)
        save_png_action.triggered.connect(self._nav_toolbar.save_figure)
        save_menu.addAction(save_png_action)

        save_menu.addSeparator()

        save_segy_action = QAction("As SEGY File...", self)
        save_segy_action.triggered.connect(self._save_segy)
        save_menu.addAction(save_segy_action)

        save_rd3_action = QAction("As MALA rd3 File...", self)
        save_rd3_action.triggered.connect(self._save_rd3)
        save_menu.addAction(save_rd3_action)

        save_rd7_action = QAction("As MALA rd7 File...", self)
        save_rd7_action.triggered.connect(self._save_rd7)
        save_menu.addAction(save_rd7_action)

        return save_menu


    def _show_context_menu(self, position) -> None:
        """Show context menu on right-click."""
        context_menu = QMenu(self)

        # Load action (single option for all formats)
        load_action = QAction("Open file...", self)
        load_action.triggered.connect(self.load_file_dialog)
        context_menu.addAction(load_action)

        # Save submenu
        context_menu.addMenu(self._create_save_menu())

        # Display Settings action
        display_settings_action = QAction("Display Settings...", self)
        display_settings_action.triggered.connect(self._show_display_settings)
        context_menu.addAction(display_settings_action)

        # Apply Filters action
        apply_filters_action = QAction("Apply Filters...", self)
        apply_filters_action.triggered.connect(self._on_filters_clicked)
        context_menu.addAction(apply_filters_action)

        # Show menu at cursor position
        context_menu.exec(self.mapToGlobal(position))


    def load_file_dialog(self) -> bool:
        """Load a seismic file (SEGY, rd3, or rd7)."""
        # Use directory of current file as default, or empty string if no file loaded
        default_dir = os.path.dirname(self.filename)

        new_filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Seismic File",
            default_dir,
            "Seismic Files (*.sgy *.segy *.rd3 *.rd7);;SEGY Files (*.sgy *.segy);;MALA Files (*.rd3 *.rd7);;All Files (*)",
        )
        if not new_filename:
            return False
        return self.load_file(new_filename)


    def load_file(self, new_filename: str) -> bool:
        new_data_file = DataFile(new_filename)
        if not new_data_file.load():
            self._show_error("Error", f"Failed to load file: {new_filename}\n{new_data_file.error}")
            return False
        self._data_file = new_data_file
        # clear previous file's cached data
        self._processed_data = None
        self._trace_cumulative_distances_meters = None
        self._trace_cumulative_distances_display = None
        self._time_first_arrival_seconds = 0.0
        self._time_axis_values_seconds = None
        self._file_view_region = None
        self._file_region_clipped = None
        self._canvas_render_region = None
        self._canvas_buffer = None
        self._depth_converted_meters = None
        self._depth_converted_display = None
        self._horizontal_indices_in_data_region = None
        self._vertical_indices_in_data_region = None
        self._amplitude_min = 0.0
        self._amplitude_max = 0.0
        self._image = None
        self._display_settings['ind_sample_time_first_arrival'] = None
        self._filter_pipeline.clear()
        self._remove_colorbar_indicator()
        self._remove_colorbar()
        ############################################################
        # complete load file
        ############################################################
        self._apply_filters() # apply filters to the data to get the processed data
        self._trace_cumulative_distances_meters = self._data_file.trace_comulative_distances_meters()

        # Calculate time range in seconds
        dt_seconds = self._data_file.time_interval_seconds
        nt = self.raw_data.shape[0]
        time_range_seconds = (nt - 1) * dt_seconds
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

        self._time_first_arrival_seconds = self._data_file.time_delay_seconds

        if self._time_first_arrival_seconds < dt_seconds or self._time_first_arrival_seconds >= time_range_seconds - dt_seconds:
            # default to 10% of the data range
            self._time_first_arrival_seconds = time_range_seconds * 0.1
        
        ind_sample_time_first_arrival = self._display_settings.get('ind_sample_time_first_arrival', None)
        if ind_sample_time_first_arrival is None:
            ind_sample_time_first_arrival = int(self._time_first_arrival_seconds / dt_seconds)
        self._update_first_arrival_sample(ind_sample_time_first_arrival)
        self._calculate_depth_converted()
        self._reset_view_file_region_to_home() # creating the canvas buffer
        self.update_status(f"Loaded: {self.filename}")
        self.canvas_render()
        return True


    def _on_draw_complete(self, event) -> None:
        """Cache colorbar background after draw completes for fast blitting."""
        if self._colorbar is None:
            return
        try:
            self._colorbar_background = self._canvas.copy_from_bbox(self._colorbar.ax.bbox)
        except Exception:
            # If copy fails, just skip caching
            self._colorbar_background = None


    def _on_resize(self, event) -> None:
        """Handle canvas resize event to maintain fixed pixel margins."""
        # Invalidate cached background since layout changes
        self._colorbar_background = None
        self._adjust_layout_with_fixed_margins()
        self._recreate_canvas_buffer()
        self._set_canvas_to_image()


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
        current_canvas_shape = self._get_canvas_buffer_shape()
        if current_canvas_shape[0] == 0 or current_canvas_shape[1] == 0:
            self._reset_view_file_region_to_home()
            return
        new_canvas_shape = self._get_pixels_shape()
        if new_canvas_shape == current_canvas_shape:
            self._canvas.draw_idle()
            return
        if self._home_mode == self.HOME_MODE_1X1:
            pixels_shape = self._get_pixels_shape()
            file_shape = self._get_file_shape()
            new_file_view_region_shape = (min(pixels_shape[0], file_shape[0]), min(pixels_shape[1], file_shape[1]))
            new_file_view_region_x0 = max(0, self._file_view_region.x0)
            new_file_view_region_y0 = max(0, self._file_view_region.y0)
            new_file_view_region_x1 = new_file_view_region_x0 + new_file_view_region_shape[1]
            new_file_view_region_y1 = new_file_view_region_y0 + new_file_view_region_shape[0]
            self._set_file_view_region(new_file_view_region_x0, new_file_view_region_y0, new_file_view_region_x1, new_file_view_region_y1)
        else:
            # keep the same file view region and the same file region clipped
            pass


    def _on_global_settings_changed(self) -> None:
        """Callback when global settings change - update the layout."""
        self._adjust_layout_with_fixed_margins()
        self._recreate_canvas_buffer()
        self._set_canvas_to_image()


    def canvas_render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._canvas_buffer is None:
            return

        self._remove_colorbar()
        self._remove_colorbar_indicator()

        self._image_ax.clear()
        self._check_canvas_buffer_shape()
        if self._canvas_buffer is not None:

            # Apply colormap settings
            colormap = self._get_colormap()

            # Display the image (should not interpolate because data is already resampled)
            self._image = self._image_ax.imshow(self._canvas_buffer, cmap=colormap, vmin=self._amplitude_min, vmax=self._amplitude_max, interpolation='none')
            self._update_image_extent_and_limits()

            # Apply other display settings
            self._check_colorbar_ax_visibility()
            self._update_file_name_in_plot(None)
            self._apply_image_four_axes_tick_settings()
            self._canvas.draw_idle()

        self._adjust_layout_with_fixed_margins()
        self._recreate_canvas_buffer()
        self._set_canvas_to_image()

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


    def _add_open_button(self) -> None:
        """Add an Open button to the NavigationToolbar."""
        # Get the standard "Open" icon from the Qt style
        open_icon = self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton)

        # Create the Open action
        open_action = QAction(open_icon, "Open", self)
        open_action.setToolTip("Open file")
        open_action.triggered.connect(self.load_file_dialog)

        # Insert the Open action at the beginning of the toolbar (before Home)
        actions = self._nav_toolbar.actions()
        if actions:
            self._nav_toolbar.insertAction(actions[0], open_action)
        else:
            self._nav_toolbar.addAction(open_action)


    def _add_filters_button(self) -> None:
        """Add a Filters button to the NavigationToolbar after the Save button."""
        # Use a standard Qt icon for filters
        filter_icon = self.style().standardIcon(self.style().StandardPixmap.SP_FileDialogDetailedView)

        # Create the Filters action
        filter_action = QAction(filter_icon, "Filters", self)
        filter_action.setToolTip("Apply Filters...")
        filter_action.triggered.connect(self._on_filters_clicked)

        # Find the Save action and insert after it with a separator
        save_action_index = None
        actions = self._nav_toolbar.actions()
        for i, action in enumerate(actions):
            if "Save" in action.text() or "save" in action.toolTip().lower():
                save_action_index = i
                break

        if save_action_index is not None and save_action_index + 1 < len(actions):
            # Insert filter action first, then custom separator before it
            next_action = actions[save_action_index + 1]
            self._nav_toolbar.insertAction(next_action, filter_action)

            # Create custom separator using QFrame for full control
            sep_frame = QFrame()
            sep_frame.setFrameShape(QFrame.Shape.VLine)
            sep_frame.setFrameShadow(QFrame.Shadow.Plain)
            sep_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

            # Style with contrasting color based on toolbar background
            bg_color = self._nav_toolbar.palette().color(self._nav_toolbar.backgroundRole())
            separator_color = "#404040" if bg_color.lightness() > 128 else "#c0c0c0"
            sep_frame.setStyleSheet(f"color: {separator_color}; margin: 0px 4px;")
            sep_frame.setFixedWidth(2)

            self._nav_toolbar.insertWidget(filter_action, sep_frame)
        else:
            # Fallback: add at the end
            self._nav_toolbar.addSeparator()
            self._nav_toolbar.addAction(filter_action)


    def _on_filters_clicked(self) -> None:
        """Open the filters dialog (non-modal)."""
        # Keep reference so dialog isn't garbage collected
        self._filters_dialog = FiltersDialog(self)
        self._filters_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._filters_dialog.show()


    def _apply_filters_and_render(self, pipeline_state: list) -> None:
        """Apply filter pipeline and update the display.

        Args:
            pipeline_state: Serialized pipeline state to apply (list of filter dicts)
        """
        before_apply_filters = self._get_file_shape()
        self._filter_pipeline.deserialize(pipeline_state)
        self._apply_filters()
        # now canvas buffer is invalidated, we need to recreate it
        after_apply_filters = self._get_file_shape()
        if before_apply_filters != after_apply_filters:
            # recreate canvas buffer according to home mode
            self._reset_view_file_region_to_home()
        else:
            # recreate canvas buffer according to the same file region clipped and file view region
            self._recreate_canvas_buffer()
        self._set_canvas_to_image()


    def _apply_filters(self) -> None:
        """Apply filter pipeline and set the processed data."""
        try:
            # if raw_data is None, it will return None without raising an exception
            self._processed_data = self._filter_pipeline.apply(self.raw_data, self.time_interval_seconds)
            assert (self._processed_data is None) == (self.raw_data is None), f'_apply_filters() {self._processed_data=} != {self.raw_data=}'
        except Exception as e:
            self._show_error("Error", f"Failed to apply filters: {e}")
            return
        self._canvas_buffer = None
        if self._processed_data is None:
            self._amplitude_min = 0.0
            self._amplitude_max = 0.0
            return
        self._amplitude_min = float(np.min(self._processed_data))
        self._amplitude_max = float(np.max(self._processed_data))


    def _replace_configure_subplots_action(self) -> None:
        """Replace the 'Configure subplots' toolbar button with Display Settings."""
        # Find the configure subplots action in the NavigationToolbar
        for action in self._nav_toolbar.actions():
            # The configure subplots action typically has "Subplots" or "Configure" in its text
            # or uses the configure_subplots method
            if action.text() == "Configure subplots" or "Subplots" in action.text():
                # Disconnect the default action
                try:
                    action.triggered.disconnect()
                except:
                    pass  # In case it's not connected

                # Connect to our Display Settings dialog
                action.triggered.connect(self._show_display_settings)

                # Change the tooltip
                action.setToolTip("Display Settings")
                break


    def _replace_home_action(self) -> None:
        """Replace the 'Home' toolbar button with custom home behavior."""
        # Find the home action in the NavigationToolbar
        for action in self._nav_toolbar.actions():
            # The home action typically has "Home" in its text or tooltip
            if action.text() == "Home" or "Home" in action.toolTip():
                # Disconnect the default action
                try:
                    action.triggered.disconnect()
                except:
                    pass  # In case it's not connected

                # Connect to our custom home handler
                action.triggered.connect(self._home)
                break


    def _replace_save_action(self) -> None:
        """Replace the 'Save' toolbar button to show save menu."""
        # Find the save action in the NavigationToolbar
        for action in self._nav_toolbar.actions():
            # The save action typically has "Save" in its text or tooltip
            if "Save" in action.text() or "Save" in action.toolTip():
                # Store reference to the action for positioning the menu
                self._save_toolbar_action = action

                # Disconnect the default action
                try:
                    action.triggered.disconnect()
                except:
                    pass  # In case it's not connected

                # Connect to our custom handler that shows the menu
                action.triggered.connect(self._show_save_menu_from_toolbar)
                break


    def _show_save_menu_from_toolbar(self) -> None:
        """Show the save menu when Save toolbar button is clicked."""
        # Create the save menu
        save_menu = self._create_save_menu()

        # Find the Save button widget to position the menu below it
        for widget in self._nav_toolbar.children():
            if hasattr(widget, 'defaultAction') and widget.defaultAction() == self._save_toolbar_action:
                # Show menu below the button
                save_menu.exec(widget.mapToGlobal(widget.rect().bottomLeft()))
                return

        # Fallback: show at toolbar position if we can't find the button
        save_menu.exec(self._nav_toolbar.mapToGlobal(self._nav_toolbar.rect().center()))


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
        if self._image is None:
            return
        # Update the image colormap
        self._image.set_cmap(self._get_colormap())
        # Invalidate cached background (colorbar appearance changed)
        self._colorbar_background = None
        # Force full draw to recache background for blitting
        self._canvas.draw()


    def _get_colormap(self) -> str:
        colormap = self._display_settings['colormap']
        if self._display_settings['flip_colormap']:
            colormap = colormap + '_r'
        return colormap

    def _set_colorbar_visibility(self, visible: bool|int):
        if isinstance(visible, int):
            visible = visible != 0
        self._display_settings['colorbar_visible'] = visible
        self._check_colorbar_ax_visibility()
        self._adjust_layout_with_fixed_margins()
        self._recreate_canvas_buffer()
        self._set_canvas_to_image()



    def _update_file_name_in_plot(self, set_value: bool|None) -> None:
        if set_value is not None:
            self._display_settings['file_name_in_plot'] = set_value
        else:
            set_value = self._display_settings['file_name_in_plot']

        if set_value:
            self._image_ax.set_title(os.path.basename(self.filename))
        else:
            self._image_ax.set_title("")


    def _apply_image_four_axes_tick_settings(self) -> None:
        """Apply tick settings and labels to all four axes surrounding the image."""

        # Clear all secondary axes to avoid stacking
        # Note: Only secondary axes (top/right) need clearing, not primary axes (bottom/left).
        # Primary axes (xaxis/yaxis) are single objects that get updated in place.
        # Secondary axes are created by secondary_xaxis()/secondary_yaxis() calls,
        # which create NEW axes objects each time, so old ones must be removed first.
        if self._image_ax is None or self._image_ax.figure is not self._fig:
            return
        if hasattr(self._image_ax, 'child_axes'):
            for child_ax in self._image_ax.child_axes[:]:  # [:] creates copy while iterating
                child_ax.remove()

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
            self._apply_single_axis_tick_settings(ax2.xaxis, top, self._display_settings['top_major_tick'], self._display_settings['top_minor_ticks'])

        # Apply bottom axis
        if bottom == AxisType.NONE:
            self._image_ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
        else:
            self._image_ax.xaxis.set_tick_params(bottom=True, labelbottom=True)
            self._apply_single_axis_tick_settings(self._image_ax.xaxis, bottom, self._display_settings['bottom_major_tick'], self._display_settings['bottom_minor_ticks'])

        # Apply left axis
        if left == AxisType.NONE:
            self._image_ax.yaxis.set_tick_params(left=False, labelleft=False)
        else:
            self._image_ax.yaxis.set_tick_params(left=True, labelleft=True)
            self._apply_single_axis_tick_settings(self._image_ax.yaxis, left, self._display_settings['left_major_tick'], self._display_settings['left_minor_ticks'])

        # Apply right axis
        if right == AxisType.NONE:
            self._image_ax.yaxis.set_tick_params(right=False, labelright=False)
            self._image_ax.secondary_yaxis('right').set_visible(False)
        else:
            ax2 = self._image_ax.secondary_yaxis('right')
            ax2.set_visible(True)
            ax2.tick_params(axis='y', right=True, labelright=True)
            self._apply_single_axis_tick_settings(ax2.yaxis, right, self._display_settings['right_major_tick'], self._display_settings['right_minor_ticks'])


    def _check_colorbar_ax_visibility(self) -> bool:
        colorbar_visible = self._display_settings['colorbar_visible'] # must check with default value True
        if not colorbar_visible or self._canvas_buffer is None or np.isnan(self._amplitude_min) or np.isnan(self._amplitude_max):
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
        if hasattr(self, '_colorbar_ax') and self._colorbar_ax is not None:
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


    def _remove_colorbar_indicator(self) -> None:
        if self._colorbar_indicator is None:
            return
        self._colorbar_indicator.set_visible(False)
        self._colorbar_indicator = None


    def _apply_single_axis_tick_settings(self, axis: plt.Axes, axis_type: AxisType, major_tick_distance: float, minor_ticks_per_major: int) -> None:
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
        # minor ticks do not have tick labels
        axis.set_minor_formatter(plt.NullFormatter()) 

        axis_vector_values_for_display = self._get_axis_values_for_display(axis_type)
        if axis_vector_values_for_display is None:
            axis_vector_indices = None
            axis_vector_values = None
        else:
            axis_vector_indices, axis_vector_values = axis_vector_values_for_display
        if axis_vector_indices is None or len(axis_vector_indices) == 0 or axis_vector_values is None or len(axis_vector_values) == 0:
            axis.set_major_locator(plt.NullLocator())
            axis.set_minor_locator(plt.NullLocator())
            return
        assert len(axis_vector_indices) == len(axis_vector_values)
        if len(axis_vector_values) == 1:
            axis.set_major_locator(plt.FixedLocator([axis_vector_values[0]]))
            axis.set_minor_locator(plt.NullLocator())
            return
        average_tick_distance = (axis_vector_values[-1] - axis_vector_values[0]) / (len(axis_vector_values) - 1)
        minor_ticks_per_major = max(minor_ticks_per_major, 1)
        major_tick_distance = max(major_tick_distance, average_tick_distance)
        minor_tick_distance = major_tick_distance / minor_ticks_per_major
        ind_minor_min = int(np.ceil(axis_vector_values[0] / minor_tick_distance))
        ind_minor_max = int(np.floor(axis_vector_values[-1] / minor_tick_distance))
        if ind_minor_max - ind_minor_min < 0:
            axis.set_major_locator(plt.NullLocator())
            axis.set_minor_locator(plt.NullLocator())
            return
        ind_all_ticks = np.arange(ind_minor_min, ind_minor_max + 1)
        all_tick_values = ind_all_ticks * minor_tick_distance
        all_tick_positions = np.interp(all_tick_values, axis_vector_values, axis_vector_indices)
        select_minors = ind_all_ticks % minor_ticks_per_major != 0
        minor_positions = all_tick_positions[select_minors]
        if len(minor_positions) > 0:
            axis.set_ticks(minor_positions, minor=True)
        else:
            axis.set_minor_locator(plt.NullLocator())
        
        major_positions = all_tick_positions[~select_minors]
        if len(major_positions) > 0:
            axis.set_ticks(major_positions)
            axis.set_ticklabels([format_value(val, 3) for val in all_tick_values[~select_minors]])
        else:
            axis.set_major_locator(plt.NullLocator())


    def _on_change_first_arrival_sample(self, ind_sample_time_first_arrival: int) -> int:
        self._update_first_arrival_sample(ind_sample_time_first_arrival)
        self._calculate_depth_converted()
        self._resample_axis_values_4_display()
        self._apply_image_four_axes_tick_settings()
        self._canvas.draw_idle()
        return self._display_settings['ind_sample_time_first_arrival'] # updated value


    def _update_first_arrival_sample(self, ind_sample_time_first_arrival: int|None) -> None:
        nt = self.raw_data.shape[0]
        if ind_sample_time_first_arrival is None:
            ind_sample_time_first_arrival = self._display_settings['ind_sample_time_first_arrival']
        ind_sample_time_first_arrival = max(0,min(ind_sample_time_first_arrival, nt - 1))
        dt_seconds = self.time_interval_seconds
        self._time_first_arrival_seconds = ind_sample_time_first_arrival * dt_seconds
        self._display_settings['ind_sample_time_first_arrival'] = ind_sample_time_first_arrival
        self._time_axis_values_seconds = np.arange(nt) * dt_seconds - self._time_first_arrival_seconds


    def _calculate_depth_converted(self) -> None:
        """
        Calculate depth for bistatic GPR using geometric correction.

        For GPR with fixed Tx-Rx antenna separation (offset), the signal travels
        diagonally from transmitter to reflector and back to receiver. The geometry
        forms a triangle where:
        - Slant distance: L = sqrt(d² + (offset/2)²)
        - Two-way time: t = 2L/v
        - Solving for depth: d = sqrt((v*t/2)² - (offset/2)²)
        """
        if self._time_axis_values_seconds is None:
            self._depth_converted_meters = None
            return
        air_velocity_m_per_s = min(self._display_settings['air_velocity_m_per_s'], C_VACUUM)
        ground_velocity_m_per_s = min(self._display_settings['ground_velocity_m_per_s'], C_VACUUM)
        offset_meters = self.offset_meters
        half_offset = 0.5 * offset_meters
        self._depth_converted_meters = np.empty_like(self._time_axis_values_seconds)
        # Above surface (negative time): antenna height above ground
        # Uses air velocity for propagation before first arrival
        # above ground depth is calculated using air velocity assuming one way travel time
        time_axis_values = self._time_axis_values_seconds - offset_meters / ground_velocity_m_per_s  # 2nd arrival corrected time axis values
        num_samples_1st_arrival = np.sum(self._time_axis_values_seconds < 0)
        num_samples_2nd_arrival = num_samples_1st_arrival + np.sum(time_axis_values[num_samples_1st_arrival:] < 0.0)
        self._depth_converted_meters[:num_samples_1st_arrival] = time_axis_values[:num_samples_1st_arrival] * air_velocity_m_per_s
        self._depth_converted_meters[num_samples_1st_arrival:num_samples_2nd_arrival] = time_axis_values[num_samples_1st_arrival:num_samples_2nd_arrival] * ground_velocity_m_per_s
        # Below surface: geometric correction for bistatic antenna configuration
        # Signal travels diagonally from Tx to reflector to Rx
        slant_distance = time_axis_values[num_samples_2nd_arrival:]*ground_velocity_m_per_s/2
        # clip to zero depth. This handles the "direct wave zone" near the surface.
        self._depth_converted_meters[num_samples_2nd_arrival:] = np.sqrt(np.maximum(slant_distance**2 - half_offset**2, 0))


    def _resample_axis_values_4_display(self) -> None:
        crop_response = self._file_region_clipped
        render_region = self._canvas_render_region
        if render_region is None or crop_response is None:
            return

        def resample_axis_values_to_display(axis_values: np.ndarray | None, file_ind0: int, file_ind1: int, pixel_ind0: int, pixel_ind1: int, display_factor: float) -> np.ndarray | None:
            if axis_values is None or len(axis_values) == 0 or file_ind0 >= file_ind1 or pixel_ind0 >= pixel_ind1 or display_factor is None or np.isnan(display_factor) or display_factor <= 0.0:
                return None
            # resample axis values to display to be the same length as the display data
            display_length = pixel_ind1 - pixel_ind0
            file_values_sliced = axis_values[file_ind0:file_ind1] * display_factor
            slice_length = len(file_values_sliced)
            display_indices = np.linspace(start=0,stop=slice_length-1,num=display_length)
            return np.interp(display_indices, np.arange(slice_length), file_values_sliced)
        
        self._trace_cumulative_distances_display = resample_axis_values_to_display(\
            axis_values=self._trace_cumulative_distances_meters,\
            file_ind0=int(crop_response.x0),\
            file_ind1=int(crop_response.x1),\
            pixel_ind0=int(render_region.x0),\
            pixel_ind1=int(render_region.x1),\
            display_factor=GlobalSettings.display_length_factor
        )
        self._time_axis_values_display = resample_axis_values_to_display(\
            axis_values=self._time_axis_values_seconds,\
            file_ind0=int(crop_response.y0),\
            file_ind1=int(crop_response.y1),\
            pixel_ind0=int(render_region.y0),\
            pixel_ind1=int(render_region.y1),\
            display_factor=self._time_display_value_factor
        )
        self._depth_converted_display = resample_axis_values_to_display(\
            axis_values=self._depth_converted_meters,\
            file_ind0=int(crop_response.y0),\
            file_ind1=int(crop_response.y1),\
            pixel_ind0=int(render_region.y0),\
            pixel_ind1=int(render_region.y1),\
            display_factor=GlobalSettings.display_length_factor
        )
        self._horizontal_indices_in_data_region = resample_axis_values_to_display(\
            axis_values=np.arange(self.raw_data.shape[1], dtype=int),\
            file_ind0=int(crop_response.x0),\
            file_ind1=int(crop_response.x1),\
            pixel_ind0=int(render_region.x0),\
            pixel_ind1=int(render_region.x1),\
            display_factor=1.0
        )
        self._vertical_indices_in_data_region = resample_axis_values_to_display(\
            axis_values=np.arange(self.raw_data.shape[0], dtype=int),\
            file_ind0=int(crop_response.y0),\
            file_ind1=int(crop_response.y1),\
            pixel_ind0=int(render_region.y0),\
            pixel_ind1=int(render_region.y1),\
            display_factor=1.0
        )


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
        if self._processed_data is None:
            self._show_error("Error", "No data to save")
            return
        data_2_save = self._processed_data.astype(np.float32)
        while True:
            # Create default filename by concatenating current filename with .file_type
            default_name = os.path.basename(self.filename) + '.' + file_type
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

            new_filename, _ = QFileDialog.getSaveFileName(self, f"Save as {label} File", default_name, f"{label} Files ({filter});;All Files (*)")
            if not new_filename:
                # User cancelled
                return

            new_data_file = DataFile(new_filename)
            new_data_file.data = data_2_save
            if new_data_file.save():
                QMessageBox.information(self, "Success", f"Saved to {new_filename}")
                break
            
            action = self._show_save_error_dialog(f"Failed to save file: {new_filename}\n{new_data_file.error}", label)
            if action == 'cancel':
                break
            if action == 'change_format':
                # Show save format menu
                self._show_save_format_menu()
            # default to retry --> continue


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
        elif clicked == cancel_btn:
            return 'cancel'
        else:
            return 'unknown'



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


