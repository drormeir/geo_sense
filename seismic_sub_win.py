import numpy as np
import os
from typing import Any

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QMenu, QToolBar, QDialog, QFormLayout, QComboBox, QDialogButtonBox, QGroupBox, QCheckBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from matplotlib.backend_bases import NavigationToolbar2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uas import UASSubWindow, UASMainWindow, auto_register

import segyio
from gprpy.toolbox.gprIO_MALA import readMALA
from gs_icon import create_gs_icon
from global_settings import GlobalSettings


class DisplaySettingsDialog(QDialog):
    """
    Dialog for configuring display axis settings for each border.
    """
    default_settings = {
        'top': 'None',
        'bottom': 'Distance',
        'left': 'Sample',
        'right': 'None',
        'colormap': 'seismic',
        'flip_colormap': False,
        'colorbar_visible': True,
        'file_name_in_plot': True,
    }
    colormap_options = ["seismic", "gray", "viridis", "plasma", "RdBu", "hot", "coolwarm", "jet"]
    vertical_options = ["None", "Sample", "Time", "Depth"]
    horizontal_options = ["None", "Distance", "Trace sample"]


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

        # Create form layout for border settings
        form_layout = QFormLayout()

        # Top and Bottom borders (horizontal)

        self.top_combo = QComboBox()
        self.top_combo.addItems(DisplaySettingsDialog.horizontal_options)
        self.top_combo.setCurrentText(current_settings['top'])
        self.top_combo.currentIndexChanged.connect(self._on_setting_changed)
        form_layout.addRow("Top border:", self.top_combo)

        self.bottom_combo = QComboBox()
        self.bottom_combo.addItems(DisplaySettingsDialog.horizontal_options)
        self.bottom_combo.setCurrentText(current_settings['bottom'])
        self.bottom_combo.currentIndexChanged.connect(self._on_setting_changed)
        form_layout.addRow("Bottom border:", self.bottom_combo)

        # Left and Right borders (vertical)

        self.left_combo = QComboBox()
        self.left_combo.addItems(DisplaySettingsDialog.vertical_options)
        self.left_combo.setCurrentText(current_settings['left'])
        self.left_combo.currentIndexChanged.connect(self._on_setting_changed)
        form_layout.addRow("Left border:", self.left_combo)

        self.right_combo = QComboBox()
        self.right_combo.addItems(DisplaySettingsDialog.vertical_options)
        self.right_combo.setCurrentText(current_settings['right'])
        self.right_combo.currentIndexChanged.connect(self._on_setting_changed)
        form_layout.addRow("Right border:", self.right_combo)

        layout.addLayout(form_layout)

        # Colormap selection
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QFormLayout()

        # Common colormaps
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(DisplaySettingsDialog.colormap_options)
        self.colormap_combo.setCurrentText(current_settings['colormap'])
        self.colormap_combo.currentIndexChanged.connect(self._on_setting_changed)
        colormap_layout.addRow("Color scheme:", self.colormap_combo)

        # Flip colormap checkbox
        self.flip_colormap_checkbox = QCheckBox("Flip colormap")
        self.flip_colormap_checkbox.setChecked(current_settings['flip_colormap'])
        self.flip_colormap_checkbox.stateChanged.connect(self._on_setting_changed)
        colormap_layout.addRow("", self.flip_colormap_checkbox)

        # Colorbar visible checkbox
        self.colorbar_visible_checkbox = QCheckBox("Show Colorbar")
        self.colorbar_visible_checkbox.setChecked(current_settings['colorbar_visible'])
        self.colorbar_visible_checkbox.stateChanged.connect(self._on_setting_changed)
        colormap_layout.addRow("", self.colorbar_visible_checkbox)

        # File name in plot checkbox
        self.file_name_in_plot_checkbox = QCheckBox("Show file name in plot")
        self.file_name_in_plot_checkbox.setChecked(current_settings['file_name_in_plot'])
        self.file_name_in_plot_checkbox.stateChanged.connect(self._on_setting_changed)
        colormap_layout.addRow("", self.file_name_in_plot_checkbox)

        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)

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
            'top': self.top_combo.currentText(),
            'bottom': self.bottom_combo.currentText(),
            'left': self.left_combo.currentText(),
            'right': self.right_combo.currentText(),
            'colormap': self.colormap_combo.currentText(),
            'flip_colormap': self.flip_colormap_checkbox.isChecked(),
            'colorbar_visible': self.colorbar_visible_checkbox.isChecked(),
            'file_name_in_plot': self.file_name_in_plot_checkbox.isChecked()
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
        self._sample_interval: float = 1.0
        self._sample_unit: str = "sample"
        self._sample_min: float = 0.0
        self._is_depth: bool = False
        self._distance_unit: str = "trace"

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

        # Create toolbar
        self._toolbar = QToolBar("Seismic Toolbar", self)
        self._setup_toolbar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._toolbar)
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
        if abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
            self._axes.set_xlim(min(x0, x1), max(x0, x1))
            self._axes.set_ylim(max(y0, y1), min(y0, y1))  # Inverted for image coordinates
            self._canvas.draw_idle()

        self._press_event = None


    def _zoom_out(self) -> None:
        """Zoom out to show full data extent."""
        if self._data is not None:
            self._axes.set_xlim(0, self._data.shape[1])
            self._axes.set_ylim(self._data.shape[0], 0)  # Inverted for image coordinates
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
            x, y = int(round(event.xdata)), int(round(event.ydata))
            hover_info = self.get_hover_info(x, y)

        amplitude = hover_info.get('amplitude', None) if hover_info is not None else None
        self._update_colorbar_indicator(amplitude)
        self._main_window.on_subwindow_hover(hover_info)


    def _axes_leave_event(self, event) -> None:
        """Handle canvas leave event and clear status bar."""
        self._update_colorbar_indicator(None)
        self._main_window.on_subwindow_hover({})


    def _update_colorbar_indicator(self, amplitude: float|None) -> None:
        self.remove_colorbar_indicator()

        """Update the horizontal indicator line on the colorbar."""
        if self._image is not None and self._colorbar is not None and amplitude is not None:
            try:
                norm_value = (amplitude - self._amplitude_min) / (self._amplitude_max - self._amplitude_min)
                norm_value = max(0.0, min(1.0, norm_value))  # Clamp to [0, 1]
                # Get the color from the colormap
                cmap = self._image.get_cmap()
                rgba = cmap(norm_value)
                # Get inverted color for visibility
                inv_color = (1.0 - rgba[0], 1.0 - rgba[1], 1.0 - rgba[2])
                # Draw new indicator line spanning the colorbar width
                self._colorbar_indicator = self._colorbar.ax.axhline(y=amplitude, color=inv_color, linewidth=2, alpha=1.0)
            except (np.linalg.LinAlgError, ValueError):
                # Colorbar axis not yet properly initialized, skip indicator
                pass
        self._canvas.draw_idle()


    def get_hover_info(self, x: int, y: int) -> dict[str, Any] | None:
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

        # Check bounds
        if not (0 <= y < self._data.shape[0] and 0 <= x < self._data.shape[1]):
            return None

        # Calculate absolute depth/time value: value = min + (sample * interval)
        absolute_value = self._sample_min + (y * self._sample_interval)

        hover_info = {
            'trace_number': x,
            'sample_number': y,
            'depth_time_value': absolute_value,
            'depth_time_unit': self._sample_unit,
            'is_depth': self._is_depth,  # True = depth, False = time
            'amplitude': self._data[y, x],
        }

        # Calculate horizontal distance
        if self._trace_coords is not None and x < len(self._trace_coords):
            if x == 0:
                horizontal_distance = 0.0
            else:
                # Calculate cumulative distance along the line
                coord_current = self._trace_coords[x]
                coord_prev = self._trace_coords[0]
                horizontal_distance = np.sqrt(
                    (coord_current[0] - coord_prev[0])**2 +
                    (coord_current[1] - coord_prev[1])**2
                )
            hover_info['horizontal_distance'] = horizontal_distance
        else:
            hover_info['horizontal_distance'] = float(x)

        hover_info['distance_unit'] = self._distance_unit

        return hover_info


    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including loaded file and color scale."""
        state = super().serialize()
        state['filename'] = self._filename
        state['display_settings'] = self._display_settings
        return state


    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including loaded file and color scale."""
        super().deserialize(state)
        if 'display_settings' in state:
            self._display_settings = state['display_settings']
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

        try:
            if file_type == 'sgy':
                with segyio.open(filename, "r", ignore_geometry=True) as f:
                    self._data = f.trace.raw[:].T

                    # Extract metadata from SEGY file
                    # Sample interval from binary header (in microseconds)
                    dt_us = f.bin[segyio.BinField.Interval]
                    if dt_us > 0:
                        self._sample_interval = dt_us / 1_000_000.0  # Convert to seconds
                        self._sample_unit = "s"
                    else:
                        self._sample_interval = 1.0
                        self._sample_unit = "sample"

                    # Check if depth or time (assume time by default for SEGY)
                    self._is_depth = False

                    # Try to extract trace coordinates from trace headers
                    try:
                        num_traces = len(f.trace)
                        coords = np.zeros((num_traces, 2))
                        for i in range(num_traces):
                            # CDP X and Y coordinates (scaled)
                            x = f.header[i][segyio.TraceField.CDP_X]
                            y = f.header[i][segyio.TraceField.CDP_Y]
                            coords[i] = [x, y]

                        # Only use coordinates if they're not all zeros
                        if np.any(coords):
                            self._trace_coords = coords
                            self._distance_unit = "m"
                        else:
                            self._trace_coords = None
                            self._distance_unit = "trace"
                    except Exception as e:
                        print(f"Error extracting trace coordinates: {e}")
                        self._trace_coords = None
                        self._distance_unit = "trace"

                    self._sample_min = 0.0

            else:
                file_base, _ = os.path.splitext(filename)
                self._data, info = readMALA(file_base)
                self._data = np.array(self._data)

                # Extract metadata from MALA header
                # Calculate sample interval (dt) from TIMEWINDOW and SAMPLES
                timewindow = float(info.get('TIMEWINDOW', 0))
                samples = int(info.get('SAMPLES', 1))
                if timewindow > 0 and samples > 0:
                    self._sample_interval = timewindow / samples
                    self._sample_unit = "ns"
                else:
                    self._sample_interval = 1.0
                    self._sample_unit = "sample"

                # MALA is typically time-based (GPR data)
                self._is_depth = False

                # Distance interval from MALA header
                distance_interval = float(info.get('DISTANCE INTERVAL', 0))
                if distance_interval > 0:
                    num_traces = self._data.shape[1]
                    # Create linear coordinates based on distance interval
                    distances = np.arange(num_traces) * distance_interval
                    self._trace_coords = np.column_stack([distances, np.zeros(num_traces)])
                    self._distance_unit = "m"
                else:
                    self._trace_coords = None
                    self._distance_unit = "trace"

                self._sample_min = 0.0

            self._filename = filename
            percentile = np.percentile(np.abs(self._data), 95)
            self._amplitude_min = -percentile
            self._amplitude_max = percentile
            ret = True
        except Exception as e:
            self._show_error("Error", f"Failed to load file:\n{str(e)}")
            self.remove_colorbar_indicator()
            self.remove_colorbar()
            self._trace_coords = None
            self._distance_unit = "trace"
            self._sample_interval = 1.0
            self._sample_unit = "sample"
            self._sample_min = 0.0
            self._is_depth = False
            self._amplitude_min = 0.0
            self._amplitude_max = 0.0
            self._image = None
            self._filename = ""
            ret = False

        self.canvas_render()
        return ret


    def canvas_render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._data is None:
            return
        self.remove_colorbar()
        self.remove_colorbar_indicator()

        self._axes.clear()

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
        if self._display_settings.get('top', 'None') != 'None':
            top_margin_px += horizontal_axes_margin_px
        if self._display_settings.get('file_name_in_plot', True):
            top_margin_px += base_vertical_margin_px

        bottom_margin_px = base_vertical_margin_px
        if self._display_settings.get('bottom', 'None') != 'None':
            bottom_margin_px += horizontal_axes_margin_px

        bottom = bottom_margin_px / height_px
        height = 1.0 - (top_margin_px + bottom_margin_px) / height_px

        left_image_margin_px = base_horizontal_margin_px
        if self._display_settings.get('left', 'None') != 'None':
            left_image_margin_px += vertical_axes_margin_px

        right_image_margin_px = base_horizontal_margin_px
        if self._display_settings.get('right', 'None') != 'None':
            right_image_margin_px += vertical_axes_margin_px

        if self._display_settings.get('colorbar_visible', True):
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

        colorbar_visible = self._display_settings.get('colorbar_visible', True)
        colormap = self._display_settings.get('colormap', 'seismic')
        flip_colormap = self._display_settings.get('flip_colormap', False)
        file_name_in_plot = self._display_settings.get('file_name_in_plot', True)
        # Add '_r' suffix to flip the colormap
        if flip_colormap:
            colormap = colormap + '_r'

        self._image = self._axes.imshow(self._data, aspect="auto", cmap=colormap, vmin=self._amplitude_min, vmax=self._amplitude_max)
        if colorbar_visible:
            if self._colorbar_axes.figure is self._fig:
                self._colorbar = self._fig.colorbar(self._image, cax=self._colorbar_axes, label="Amplitude")

        if file_name_in_plot:
            self._axes.set_title(os.path.basename(self._filename))

        # Get current settings
        top = self._display_settings.get('top', 'None')
        bottom = self._display_settings.get('bottom', 'Distance')
        left = self._display_settings.get('left', 'Sample')
        right = self._display_settings.get('right', 'None')


        # remove top and bottom labels
        self._axes.set_xlabel(None)
        self._axes.set_ylabel(None)
        self._axes.secondary_xaxis('top').set_visible(False)

        # Apply top axis
        if top == 'None':
            self._axes.xaxis.set_tick_params(top=False, labeltop=False)
            self._axes.secondary_xaxis('top').set_visible(False)
        else:
            ax2 = self._axes.secondary_xaxis('top')
            ax2.set_visible(True)
            ax2.tick_params(axis='x', top=True, labeltop=True)
            if top == 'Distance':
                ax2.set_xlabel("Distance" if self._distance_unit == "m" else "Trace Number")
            elif top == 'Trace sample':
                ax2.set_xlabel("Trace Number")

        # Apply bottom axis
        if bottom == 'None':
            self._axes.xaxis.set_tick_params(bottom=False, labelbottom=False)
        else:
            self._axes.xaxis.set_tick_params(bottom=True, labelbottom=True)
            if bottom == 'Distance':
                self._axes.set_xlabel("Distance" if self._distance_unit == "m" else "Trace Number")
            elif bottom == 'Trace sample':
                self._axes.set_xlabel("Trace Number")

        # Apply left axis
        if left == 'None':
            self._axes.yaxis.set_tick_params(left=False, labelleft=False)
        else:
            self._axes.yaxis.set_tick_params(left=True, labelleft=True)

            if left == 'Sample':
                self._axes.set_ylabel("Sample Number")
            elif left == 'Time':
                if self._sample_unit != "sample":
                    # Create labels for time values
                    num_samples = self._data.shape[0]
                    tick_positions = self._axes.get_yticks()
                    # Only keep valid tick positions and create matching labels
                    valid_ticks = [pos for pos in tick_positions if 0 <= pos < num_samples]
                    tick_labels = [f"{self._sample_min + pos * self._sample_interval:.2f}" for pos in valid_ticks]
                    self._axes.set_yticks(valid_ticks)
                    self._axes.set_yticklabels(tick_labels)
                    self._axes.set_ylabel(f"Time ({self._sample_unit})")
                else:
                    self._axes.set_ylabel("Sample Number")
            elif left == 'Depth':
                if self._is_depth and self._sample_unit != "sample":
                    # Create labels for depth values
                    num_samples = self._data.shape[0]
                    tick_positions = self._axes.get_yticks()
                    # Only keep valid tick positions and create matching labels
                    valid_ticks = [pos for pos in tick_positions if 0 <= pos < num_samples]
                    tick_labels = [f"{self._sample_min + pos * self._sample_interval:.2f}" for pos in valid_ticks]
                    self._axes.set_yticks(valid_ticks)
                    self._axes.set_yticklabels(tick_labels)
                    self._axes.set_ylabel(f"Depth ({self._sample_unit})")
                else:
                    self._axes.set_ylabel("Sample Number")

        # Apply right axis
        if right == 'None':
            self._axes.yaxis.set_tick_params(right=False, labelright=False)
            self._axes.secondary_yaxis('right').set_visible(False)
        else:
            ax2 = self._axes.secondary_yaxis('right')
            ax2.set_visible(True)
            ax2.tick_params(axis='y', right=True, labelright=True)
            if right == 'Sample':
                # Use secondary y-axis for sample numbers on right
                ax2.set_ylabel("Sample Number")
            elif right == 'Time':
                if self._sample_unit != "sample":
                    ax2.set_ylabel(f"Time ({self._sample_unit})")
                else:
                    ax2.set_ylabel("Sample Number")
            elif right == 'Depth':
                if self._is_depth and self._sample_unit != "sample":
                    ax2.set_ylabel(f"Depth ({self._sample_unit})")
                else:
                    ax2.set_ylabel("Sample Number")


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

