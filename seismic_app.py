"""
Seismic Viewer Application using UAS Framework.

A seismic data visualization application that demonstrates the UAS framework
with SEGY file support using matplotlib for rendering.

Run with: python seismic_app.py
"""

from typing import Any
import sys
import os

import numpy as np
import segyio
from gprpy.toolbox.gprIO_MALA import readMALA

from PySide6.QtWidgets import (
    QVBoxLayout,
    QApplication,
    QFileDialog,
    QMessageBox,
    QMenu,
)

from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from uas import (
    UASSubWindow,
    UASMainWindow,
    FactoryRegistry,
    UASApplication,
    auto_register,
    SessionManager,
)


class SeismicCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for displaying seismic data."""

    def __init__(self, parent=None) -> None:
        fig = Figure(figsize=(10, 6))
        super().__init__(fig)
        self._axes = fig.add_subplot(111)
        self._data: np.ndarray | None = None
        self._filename: str = ""
        self._vmin: float = 0.0
        self._vmax: float = 0.0
        self._colorbar = None
        self._image = None
        self._colorbar_indicator: Line2D | None = None

        # Metadata fields
        self._trace_coords: np.ndarray | None = None
        self._sample_interval: float = 1.0
        self._sample_unit: str = "sample"
        self._sample_min: float = 0.0
        self._is_depth: bool = False
        self._distance_unit: str = "trace"

        # Connect mouse motion event
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("axes_leave_event", self._on_axes_leave)

    @property
    def axes(self):
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

    def load_file(self, filename: str, file_type: str) -> bool:
        if file_type not in ['sgy', 'rd3', 'rd7']:
            QMessageBox.critical(None, "Error", "Invalid file type")
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
            self._vmin = -percentile
            self._vmax = percentile
            ret = True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load file:\n{str(e)}")
            ret = False

        self._render()

        return ret


    def _render(self) -> None:
        """Render the seismic data to the canvas."""
        if self._data is None:
            return

        # Remove old colorbar if exists
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        self._axes.clear()
        self._image = self._axes.imshow(
            self._data,
            aspect="auto",
            cmap="seismic",
            vmin=self._vmin,
            vmax=self._vmax,
        )
        self._colorbar = self.figure.colorbar(self._image, ax=self._axes, label="Amplitude")
        self._colorbar_indicator = None  # Reset indicator on re-render
        self._axes.set_xlabel("Trace Number")
        self._axes.set_ylabel("Sample Number")
        self._axes.set_title(os.path.basename(self._filename))
        self.figure.tight_layout()
        self.draw()


    def _on_mouse_move(self, event) -> None:
        """Handle mouse motion to show amplitude indicator on colorbar."""
        if (
            self._data is None
            or self._colorbar is None
            or event.inaxes != self._axes
        ):
            self._hide_colorbar_indicator()
            return

        # Get pixel coordinates
        x, y = int(round(event.xdata)), int(round(event.ydata))

        # Check bounds
        if not (0 <= y < self._data.shape[0] and 0 <= x < self._data.shape[1]):
            self._hide_colorbar_indicator()
            return

        # Get amplitude at cursor position
        amplitude = self._data[y, x]

        # Update colorbar indicator
        self._update_colorbar_indicator(amplitude)


    def _on_axes_leave(self, event) -> None:
        """Hide colorbar indicator when mouse leaves the axes."""
        self._hide_colorbar_indicator()


    def _get_inverted_color(self, amplitude: float) -> tuple[float, float, float]:
        """Get the inverted color for a given amplitude value."""
        # Normalize amplitude to [0, 1] range
        norm_value = (amplitude - self._vmin) / (self._vmax - self._vmin)
        norm_value = max(0.0, min(1.0, norm_value))  # Clamp to [0, 1]

        # Get the color from the colormap
        cmap = self._image.get_cmap()
        rgba = cmap(norm_value)

        # Invert RGB components
        return (1.0 - rgba[0], 1.0 - rgba[1], 1.0 - rgba[2])


    def _update_colorbar_indicator(self, amplitude: float) -> None:
        """Update the horizontal indicator line on the colorbar."""
        if self._colorbar is None or self._image is None:
            return

        cbar_ax = self._colorbar.ax

        # Remove old indicator
        if self._colorbar_indicator is not None:
            self._colorbar_indicator.remove()
            self._colorbar_indicator = None

        # Get inverted color for visibility
        inv_color = self._get_inverted_color(amplitude)

        # Draw new indicator line spanning the colorbar width
        self._colorbar_indicator = cbar_ax.axhline(
            y=amplitude, color=inv_color, linewidth=2, alpha=1.0
        )
        self.draw_idle()


    def _hide_colorbar_indicator(self) -> None:
        """Hide the colorbar indicator line."""
        if self._colorbar_indicator is not None:
            self._colorbar_indicator.remove()
            self._colorbar_indicator = None
            self.draw_idle()


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
        self._canvas: SeismicCanvas | None = None
        super().__init__(main_window, parent)


    @property
    def canvas(self) -> SeismicCanvas:
        """Get the seismic canvas."""
        return self._canvas


    def on_create(self) -> None:
        """Set up the seismic display canvas."""
        self.title = "Seismic View"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = SeismicCanvas(self)
        layout.addWidget(self._canvas)

        self.setMinimumSize(400, 300)

        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect canvas hover events to propagate to main window
        self._canvas.mpl_connect("motion_notify_event", self._on_canvas_hover)
        self._canvas.mpl_connect("axes_leave_event", self._on_canvas_leave)


    def _on_canvas_hover(self, event) -> None:
        """Handle canvas hover event and propagate to main window."""
        if event.inaxes != self._canvas.axes or self._canvas.data is None:
            return

        # Get pixel coordinates
        x, y = int(round(event.xdata)), int(round(event.ydata))

        # Get hover info from canvas
        hover_info = self._canvas.get_hover_info(x, y)

        if hover_info and self._main_window:
            # Propagate to main window
            self._main_window.on_subwindow_hover(hover_info)


    def _on_canvas_leave(self, event) -> None:
        """Handle canvas leave event and clear status bar."""
        if self._main_window:
            self._main_window.on_subwindow_hover({})


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
        success = self._canvas.load_file(filename, file_type)
        if success:
            self.title = os.path.basename(filename)
            self.update_status(f"Loaded: {filename}")
        return success


    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including loaded file and color scale."""
        state = super().serialize()
        if self._canvas:
            state["canvas_state"] = {
                "filename": self._canvas.filename,
            }
        return state


    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including loaded file and color scale."""
        super().deserialize(state)
        if "canvas_state" in state and self._canvas:
            canvas_state = state["canvas_state"]
            filename = canvas_state.get("filename", "")
            self.load_file(filename)


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


    def _show_context_menu(self, position) -> None:
        """Show context menu on right-click."""
        context_menu = QMenu(self)

        # Load action (single option for all formats)
        load_action = QAction("Load...", self)
        load_action.triggered.connect(self._load_file)
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

    def _load_file(self) -> None:
        """Load a seismic file (SEGY, rd3, or rd7)."""
        # Use directory of current file as default, or empty string if no file loaded
        default_dir = ""
        if self._canvas.filename:
            default_dir = os.path.dirname(self._canvas.filename)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Seismic File",
            default_dir,
            "Seismic Files (*.sgy *.segy *.rd3 *.rd7);;SEGY Files (*.sgy *.segy);;MALA Files (*.rd3 *.rd7);;All Files (*)",
        )
        if not filename:
            return
        self.load_file(filename)

    def _save_segy(self) -> None:
        """Save current view as SEGY file with retry logic."""
        while True:
            # Create default filename by concatenating current filename with .sgy
            default_name = ""
            if self._canvas.filename:
                default_name = self._canvas.filename + ".sgy"

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
            success, error_msg = self._canvas.save_file(filename, 'sgy')

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
            if self._canvas.filename:
                default_name = self._canvas.filename + ".rd3"

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
            success, error_msg = self._canvas.save_file(filename, 'rd3')

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
            if self._canvas.filename:
                default_name = self._canvas.filename + ".rd7"

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
            success, error_msg = self._canvas.save_file(filename, 'rd7')

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


@auto_register
class SeismicMainWindow(UASMainWindow):
    """
    Main window for the Seismic Viewer application.

    Purpose:
        Provides the main application window with menus for opening SEGY files
        and managing seismic subwindows.

    Flow:
        1. Sets up File menu with Open SEGY and session management options
        2. Opens SEGY files into new SeismicSubWindow instances
        3. Supports MDI/Tabbed display modes for multiple seismic views
    """

    type_name = "seismic_main"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Seismic Viewer")

    def _setup_menus(self) -> None:
        """Set up the menu bar with File menu and seismic-specific actions."""
        super()._setup_menus()

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        menubar.insertMenu(menubar.actions()[0], file_menu)

        open_action = QAction("&Open SEGY...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_segy_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_session)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self._quit_app)
        file_menu.addAction(quit_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        agc_action = QAction("Apply &AGC", self)
        agc_action.triggered.connect(self._apply_agc)
        tools_menu.addAction(agc_action)

        # Setup toolbar
        self._setup_toolbar()

    def _setup_toolbar(self) -> None:
        """Set up the toolbar with quick access buttons."""
        toolbar = self.create_toolbar("Main Toolbar")
        toolbar.addAction("Open", self._open_segy_file)

    def _open_segy_file(self) -> None:
        """Open a seismic file (SEGY, rd3, or rd7) and create a new seismic subwindow."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Seismic File",
            "",
            "Seismic Files (*.sgy *.segy *.rd3 *.rd7);;SEGY Files (*.sgy *.segy);;MALA Files (*.rd3 *.rd7);;All Files (*)",
        )

        if filename:
            cls = FactoryRegistry.get_instance().get_subwindow_class("seismic")
            subwindow = cls.create(self)
            if subwindow.load_file(filename):
                self.add_subwindow(subwindow)
            else:
                subwindow.deleteLater()

    def _save_session(self) -> None:
        """Save the current session to disk."""
        path = SessionManager.get_instance().save()
        self._status_bar.showMessage(f"Session saved to {path}", 3000)

    def _quit_app(self) -> None:
        """Save session and quit the application."""
        session = SessionManager.get_instance()
        session.save()
        session.auto_save_enabled = False
        QApplication.quit()

    def _apply_agc(self) -> None:
        """Apply AGC to the active seismic subwindow (placeholder)."""
        QMessageBox.information(self, "AGC", "AGC function not yet implemented")

    def on_subwindow_hover(self, hover_info: dict[str, Any]) -> None:
        """
        Update status bar with hover information from a subwindow.

        Args:
            hover_info: Dictionary containing hover data with keys:
                - trace_number: int
                - sample_number: int
                - depth_time_value: float (absolute value including offset)
                - horizontal_distance: float
                - depth_time_unit: str ('s', 'ms', 'ns', 'm', etc.)
                - distance_unit: str ('m', 'km', 'trace', etc.)
                - is_depth: bool (True for depth, False for time)
        """
        if not hover_info:
            self._status_bar.clearMessage()
            return

        # Format the status message
        parts = []

        if 'trace_number' in hover_info:
            parts.append(f"Trace: {hover_info['trace_number']}")

        if 'sample_number' in hover_info:
            parts.append(f"Sample: {hover_info['sample_number']}")

        if 'horizontal_distance' in hover_info and 'distance_unit' in hover_info:
            parts.append(f"Distance: {hover_info['horizontal_distance']:.2f} {hover_info['distance_unit']}")

        if 'depth_time_value' in hover_info and 'depth_time_unit' in hover_info:
            is_depth = hover_info.get('is_depth', False)
            label = "Depth" if is_depth else "Time"
            parts.append(f"{label}: {hover_info['depth_time_value']:.3f} {hover_info['depth_time_unit']}")

        status_message = " | ".join(parts)
        self._status_bar.showMessage(status_message)


def main() -> int:
    """Run the Seismic Viewer application."""
    app = UASApplication("Seismic Viewer")
    return app.run(default_main_window_type="seismic_main")


if __name__ == "__main__":
    sys.exit(main())
