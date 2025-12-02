"""
Seismic Viewer Application using UAS Framework.

A seismic data visualization application that demonstrates the UAS framework
with SEGY file support using matplotlib for rendering.

Run with: python seismic_app.py

Command-line options:
  --test-mode          Run in test mode (minimal GUI, faster startup)
  --auto-exit SECONDS  Automatically exit after N seconds (for testing)
  --no-session         Don't load or save session state
"""

from typing import Any
import sys
import argparse

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import QTimer


from uas import (
    UASMainWindow,
    UASApplication,
    auto_register,
    SessionManager,
)

from gs_icon import create_gs_icon
from seismic_sub_win import SeismicSubWindow
from global_settings import GlobalSettings


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
        self.setWindowIcon(create_gs_icon())

    def _setup_menus(self) -> None:
        """Set up the menu bar with File menu and seismic-specific actions."""
        super()._setup_menus()

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        menubar.insertMenu(menubar.actions()[0], file_menu)

        open_action = QAction("&Open Seismic File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_seismic_file)
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

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        GlobalSettings.create_action(settings_menu, parent=self)

        # Setup toolbar
        self._setup_toolbar()


    def _setup_toolbar(self) -> None:
        """Set up the toolbar with quick access buttons."""
        toolbar = self.create_toolbar("Main Toolbar")
        toolbar.addAction("Open", self.open_seismic_file)


    def open_seismic_file(self) -> None:
        SeismicSubWindow.create_from_load_file(self)


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

        if 'horizontal_distance' in hover_info and 'distance_unit' in hover_info:
            parts.append(f"Distance: {hover_info['horizontal_distance']:.2f} {hover_info['distance_unit']}")

        if 'sample_number' in hover_info:
            parts.append(f"Sample: {hover_info['sample_number']}")

        if 'depth_time_value' in hover_info and 'depth_time_unit' in hover_info:
            is_depth = hover_info.get('is_depth', False)
            label = "Depth" if is_depth else "Time"
            parts.append(f"{label}: {hover_info['depth_time_value']:.3f} {hover_info['depth_time_unit']}")

        status_message = " | ".join(parts)
        self._status_bar.showMessage(status_message)



def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    epilog_text = """
Examples:
  %(prog)s                                    # Normal startup
  %(prog)s --no-session                       # Start fresh without loading previous session
  %(prog)s --auto-exit 3                      # Exit automatically after 3 seconds (testing)
  %(prog)s --test-mode --no-session           # Test mode without session state
  %(prog)s --auto-exit 2 --no-session         # Quick startup test

Notes:
  --auto-exit is useful for automated testing to verify the application starts correctly
  --no-session is useful for testing default behavior without saved state
  --test-mode is reserved for future testing optimizations
"""

    parser = argparse.ArgumentParser(
        description="Seismic Viewer - A seismic data visualization application",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (minimal GUI, faster startup for automated testing)",
    )

    parser.add_argument(
        "--auto-exit",
        type=float,
        metavar="SECONDS",
        help="Automatically exit after N seconds (useful for startup tests)",
    )

    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Don't load or save session state (start fresh)",
    )

    return parser.parse_args()


def main() -> int:
    """Run the Seismic Viewer application."""
    args = parse_arguments()

    app = UASApplication("Seismic Viewer")

    # Register global settings for session persistence (unless --no-session)
    if not args.no_session:
        GlobalSettings.register()

    # Setup auto-exit timer if requested (for testing)
    if args.auto_exit:
        print(f"Auto-exit scheduled in {args.auto_exit} seconds")
        QTimer.singleShot(int(args.auto_exit * 1000), QApplication.quit)

    return app.run(default_main_window_type="seismic_main")


if __name__ == "__main__":
    sys.exit(main())
