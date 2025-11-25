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
    QPushButton,
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

    def load_segy(self, filename: str) -> bool:
        """Load and display a SEGY file. Returns True on success."""
        try:
            with segyio.open(filename, "r", ignore_geometry=True) as f:
                self._data = f.trace.raw[:].T
                self._filename = filename
                percentile = np.percentile(np.abs(self._data), 95)
                self._vmin = -percentile
                self._vmax = percentile
                self._render()
                return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load file:\n{str(e)}")
            return False

    def load_mala(self, filename: str) -> bool:
        """Load and display a MALA rd3/rd7 file. Returns True on success."""
        try:
            # readMALA expects filename without extension
            file_base, _ = os.path.splitext(filename)
            self._data, info = readMALA(file_base)
            self._data = np.array(self._data)
            self._filename = filename
            percentile = np.percentile(np.abs(self._data), 95)
            self._vmin = -percentile
            self._vmax = percentile
            self._render()
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load file:\n{str(e)}")
            return False

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

    def set_color_scale(self, vmin: float, vmax: float) -> None:
        """Set the color scale limits and re-render."""
        self._vmin = vmin
        self._vmax = vmax
        self._render()

    def save_segy(self, filename: str) -> tuple[bool, str]:
        """
        Save current data to a SEGY file.

        Returns: (success: bool, error_message: str)
        """
        if self._data is None:
            return False, "No data to save"

        try:
            # Create a minimal SEGY file with current data
            spec = segyio.spec()
            spec.samples = range(self._data.shape[0])
            spec.tracecount = self._data.shape[1]
            spec.format = 1  # 4-byte IBM float

            with segyio.create(filename, spec) as f:
                for i, trace in enumerate(self._data.T):
                    f.trace[i] = trace

            return True, ""
        except Exception as e:
            return False, str(e)

    def save_mala_rd3(self, filename: str) -> tuple[bool, str]:
        """
        Save current data to a MALA rd3 file.

        Returns: (success: bool, error_message: str)
        """
        if self._data is None:
            return False, "No data to save"

        try:
            # Save as rd3 (16-bit format)
            file_base, _ = os.path.splitext(filename)
            data_file = file_base + '.rd3'

            # Convert to int16 range
            data_normalized = self._data / np.max(np.abs(self._data))
            data_int16 = (data_normalized * 32767).astype(np.int16)

            # Save binary data
            data_int16.T.tofile(data_file)

            return True, ""
        except Exception as e:
            return False, str(e)

    def save_mala_rd7(self, filename: str) -> tuple[bool, str]:
        """
        Save current data to a MALA rd7 file.

        Returns: (success: bool, error_message: str)
        """
        if self._data is None:
            return False, "No data to save"

        try:
            # Save as rd7 (32-bit format)
            file_base, _ = os.path.splitext(filename)
            data_file = file_base + '.rd7'

            # Save as float32
            data_float32 = self._data.astype(np.float32)
            data_float32.T.tofile(data_file)

            return True, ""
        except Exception as e:
            return False, str(e)

    def get_state(self) -> dict[str, Any]:
        """Get canvas state for serialization."""
        return {
            "filename": self._filename,
            "vmin": float(self._vmin),
            "vmax": float(self._vmax),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore canvas state from serialization."""
        if "filename" in state and state["filename"]:
            self._vmin = state.get("vmin", 0.0)
            self._vmax = state.get("vmax", 0.0)
            # Load the file - this will use stored vmin/vmax after load
            filename = state["filename"]
            if os.path.exists(filename):
                file_ext = os.path.splitext(filename)[1].lower()
                try:
                    if file_ext in ['.rd3', '.rd7']:
                        # Load MALA file
                        file_base, _ = os.path.splitext(filename)
                        self._data, info = readMALA(file_base)
                        self._data = np.array(self._data)
                        self._filename = filename
                    else:
                        # Load SEGY file
                        with segyio.open(filename, "r", ignore_geometry=True) as f:
                            self._data = f.trace.raw[:].T
                            self._filename = filename

                    # Use stored vmin/vmax if available, otherwise compute
                    if self._vmin == 0.0 and self._vmax == 0.0:
                        percentile = np.percentile(np.abs(self._data), 95)
                        self._vmin = -percentile
                        self._vmax = percentile
                    self._render()
                except Exception:
                    pass  # Silently fail on restore


@auto_register
class SeismicSubWindow(UASSubWindow):
    """
    Subwindow for displaying seismic SEGY data.

    Purpose:
        Displays seismic data from SEGY files using matplotlib with a seismic
        colormap. Supports color scale adjustment and session persistence.

    Flow:
        1. Initialize with empty canvas
        2. Load SEGY file via load_file() method
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

    def load_file(self, filename: str) -> bool:
        """Load a seismic file (SEGY, rd3, or rd7) into this subwindow."""
        file_ext = os.path.splitext(filename)[1].lower()

        success = False
        if file_ext in ['.rd3', '.rd7']:
            success = self._canvas.load_mala(filename)
        else:
            success = self._canvas.load_segy(filename)

        if success:
            self.title = os.path.basename(filename)
            self.update_status(f"Loaded: {filename}")
            return True
        return False

    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including loaded file and color scale."""
        state = super().serialize()
        if self._canvas:
            state["canvas_state"] = self._canvas.get_state()
        return state

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including loaded file and color scale."""
        super().deserialize(state)
        if "canvas_state" in state and self._canvas:
            self._canvas.set_state(state["canvas_state"])

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
        if filename:
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
            success, error_msg = self._canvas.save_segy(filename)

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
            success, error_msg = self._canvas.save_mala_rd3(filename)

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
            success, error_msg = self._canvas.save_mala_rd7(filename)

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


def main() -> int:
    """Run the Seismic Viewer application."""
    app = UASApplication("Seismic Viewer")
    return app.run(default_main_window_type="seismic_main")


if __name__ == "__main__":
    sys.exit(main())
