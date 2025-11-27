import numpy as np
import os
from typing import Any

from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QFileDialog, QMenu
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from uas import UASSubWindow, UASMainWindow, auto_register

import segyio
from gprpy.toolbox.gprIO_MALA import readMALA


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
        self._image: plt.Image | None = None
        self._colorbar_indicator: Line2D | None = None

        # Metadata fields
        self._trace_coords: np.ndarray | None = None
        self._sample_interval: float = 1.0
        self._sample_unit: str = "sample"
        self._sample_min: float = 0.0
        self._is_depth: bool = False
        self._distance_unit: str = "trace"

        self.title = "Seismic View"
        self._fig = Figure(figsize=(10, 6))
        self._axes = self._fig.add_subplot(111)

        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._canvas)

        self.setMinimumSize(400, 300)

        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect canvas hover events to propagate to main window
        self._canvas.mpl_connect("axes_leave_event", self._axes_leave_event)
        self._canvas.mpl_connect("motion_notify_event", self._motion_notify_event)


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
        amplitude = None
        hover_info = None
        if event is not None and event.xdata is not None and event.ydata is not None:
            # Get pixel coordinates
            x, y = int(round(event.xdata)), int(round(event.ydata))
            hover_info = self.get_hover_info(x, y)
            data = self._data
            if data is not None:
                if (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
                    amplitude = data[y, x]
        self._update_colorbar_indicator(amplitude)
        self._main_window.on_subwindow_hover(hover_info)


    def _axes_leave_event(self, event) -> None:
        """Handle canvas leave event and clear status bar."""
        self._update_colorbar_indicator(None)
        self._main_window.on_subwindow_hover({})


    def _update_colorbar_indicator(self, amplitude: float|None) -> None:
        # Remove old indicator
        if self._colorbar_indicator is not None:
            self._colorbar_indicator.remove()
            self._colorbar_indicator = None

        """Update the horizontal indicator line on the colorbar."""
        if self._colorbar is not None and amplitude is not None:
            norm_value = (amplitude - self._amplitude_min) / (self._amplitude_max - self._amplitude_min)
            norm_value = max(0.0, min(1.0, norm_value))  # Clamp to [0, 1]
            # Get the color from the colormap
            cmap = self._image.get_cmap()
            rgba = cmap(norm_value)
            # Get inverted color for visibility
            inv_color = (1.0 - rgba[0], 1.0 - rgba[1], 1.0 - rgba[2])
            # Draw new indicator line spanning the colorbar width
            self._colorbar_indicator = self._colorbar.ax.axhline(y=amplitude, color=inv_color, linewidth=2, alpha=1.0)
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
        return state


    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including loaded file and color scale."""
        super().deserialize(state)
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
            QMessageBox.critical(self, "Error", "Invalid file")
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
            QMessageBox.critical(self, "Error", "Invalid file type")
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
                    except:
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
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            ret = False

        self.canvas_render()
        return ret


    def canvas_render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._data is None:
            return

        # Remove old colorbar if exists
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._axes.clear()
        self._image = self._axes.imshow(self._data, aspect="auto", cmap="seismic", vmin=self._amplitude_min, vmax=self._amplitude_max)
        self._colorbar = self._fig.colorbar(self._image, ax=self._axes, label="Amplitude")
        self._colorbar_indicator = None  # Reset indicator on re-render
        self._axes.set_xlabel("Trace Number")
        self._axes.set_ylabel("Sample Number")
        self._axes.set_title(os.path.basename(self._filename))
        self._fig.tight_layout()
        self._canvas.draw_idle()


    def _save_segy(self) -> None:
        """Save current view as SEGY file with retry logic."""
        while True:
            # Create default filename by concatenating current filename with .sgy
            default_name = ""
            if self.filename:
                default_name = self.filename + ".sgy"

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save as SEGY File",
                default_name,
                "SEGY Files (*.sgy *.segy);;All Files (*)",
            )

            if not filename:
                # User cancelled
                return

            # Attempt to save
            success, error_msg = self.save_file(filename, 'sgy')

            if success:
                QMessageBox.information(self, "Success", f"Saved to {filename}")
                return

            # Show error dialog with retry options
            action = self._show_save_error_dialog(error_msg, "SEGY")

            if action == 'retry':
                continue  # Loop again to retry
            elif action == 'change_format':
                # Show save format menu
                self._show_save_format_menu()
                return
            else:
                # Cancel
                return


    def _save_rd3(self) -> None:
        """Save current view as MALA rd3 file with retry logic."""
        while True:
            # Create default filename by concatenating current filename with .rd3
            default_name = ""
            if self.filename:
                default_name = self.filename + ".rd3"

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save as MALA rd3 File",
                default_name,
                "MALA rd3 Files (*.rd3);;All Files (*)",
            )

            if not filename:
                # User cancelled
                return

            # Attempt to save
            success, error_msg = self.save_file(filename, 'rd3')

            if success:
                QMessageBox.information(self, "Success", f"Saved to {filename}")
                return

            # Show error dialog with retry options
            action = self._show_save_error_dialog(error_msg, "MALA rd3")

            if action == 'retry':
                continue  # Loop again to retry
            elif action == 'change_format':
                # Show save format menu
                self._show_save_format_menu()
                return
            else:
                # Cancel
                return

    def _save_rd7(self) -> None:
        """Save current view as MALA rd7 file with retry logic."""
        while True:
            # Create default filename by concatenating current filename with .rd7
            default_name = ""
            if self.filename:
                default_name = self.filename + ".rd7"

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save as MALA rd7 File",
                default_name,
                "MALA rd7 Files (*.rd7);;All Files (*)",
            )

            if not filename:
                # User cancelled
                return

            # Attempt to save
            success, error_msg = self.save_file(filename, 'rd7')

            if success:
                QMessageBox.information(self, "Success", f"Saved to {filename}")
                return

            # Show error dialog with retry options
            action = self._show_save_error_dialog(error_msg, "MALA rd7")

            if action == 'retry':
                continue  # Loop again to retry
            elif action == 'change_format':
                # Show save format menu
                self._show_save_format_menu()
                return
            else:
                # Cancel
                return


    def save_file(self, filename: str, file_type: str) -> tuple[bool, str]:
        """
        Save current data to a file.

        Returns: (success: bool, error_message: str)
        """
        if self._data is None:
            return False, "No data to save"

        if file_type not in ['sgy', 'rd3', 'rd7']:
            return False, "Invalid file type"
        
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
        except Exception as e:
            return False, str(e)
        finally:
            return True, ""



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

