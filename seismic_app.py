"""
Seismic Viewer Application using UAS Framework.

A seismic data visualization application that demonstrates the UAS framework
with SEGY file support using matplotlib for rendering.

Run with: python seismic_app.py

Command-line options:
  --test-mode            Run in test mode (minimal GUI, faster startup)
  --auto-exit SECONDS    Automatically exit after N seconds (for testing)
  --session-mode MODE    Session mode: -1=no read/write, 0=write only, 1=normal (default)
  --screenshot PATH      Save screenshot to PATH after startup (for debugging GUI)
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
from global_settings import GlobalSettings, UnitSystem


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
    _pending_screenshot_path = None  # Class variable for passing screenshot path
    _pending_auto_exit_seconds = None  # Class variable for auto-exit timer

    def __init__(self, parent=None, screenshot_path: str = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Seismic Viewer")
        self.setWindowIcon(create_gs_icon())

        # Check for pending screenshot path from command line
        self._screenshot_path = screenshot_path or self.__class__._pending_screenshot_path
        self.__class__._pending_screenshot_path = None  # Clear after use

        # Check for pending auto-exit timer
        self._auto_exit_seconds = self.__class__._pending_auto_exit_seconds
        self.__class__._pending_auto_exit_seconds = None  # Clear after use

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

    def showEvent(self, event) -> None:
        """Handle window show event - take screenshot and setup auto-exit if requested."""
        super().showEvent(event)

        if self._screenshot_path:
            # Schedule screenshot after a short delay to ensure window is fully rendered
            QTimer.singleShot(500, self._take_screenshot)

        if self._auto_exit_seconds:
            # Schedule auto-exit timer (must be done after event loop starts)
            print(f"Auto-exit scheduled in {self._auto_exit_seconds} seconds")
            QTimer.singleShot(int(self._auto_exit_seconds * 1000), QApplication.quit)

    def _take_screenshot(self) -> None:
        """Take a screenshot of the main window and save it."""
        if not self._screenshot_path:
            return

        try:
            # Grab the window contents
            pixmap = self.grab()

            # Save to file
            if pixmap.save(self._screenshot_path):
                print(f"✓ Screenshot saved to: {self._screenshot_path}")
            else:
                print(f"✗ Failed to save screenshot to: {self._screenshot_path}")
        except Exception as e:
            print(f"✗ Screenshot error: {e}")

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
                - amplitude: float
        """
        if not hover_info:
            self._status_bar.clearMessage()
            return

        # Format the status message
        parts = []

        if 'trace_number' in hover_info:
            parts.append(f"Trace: {hover_info['trace_number']}")

        distance = hover_info.get('distance', None)
        if distance is not None:
            parts.append(f"Distance: {distance:.2f} [{GlobalSettings.display_length_unit}]")

        if 'sample_number' in hover_info:
            parts.append(f"Sample: {hover_info['sample_number']}")

        if 'z_value' in hover_info and 'z_units' in hover_info:
            label = "Depth" if hover_info.get('is_depth', False) else "Time"
            parts.append(f"{label}: {hover_info['z_value']:.3f} [{hover_info['z_units']}]")

        if 'amplitude' in hover_info:
            parts.append(f"Amplitude: {hover_info['amplitude']:.3f}")

        status_message = " | ".join(parts)
        self._status_bar.showMessage(status_message)



def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    epilog_text = """
Examples:
  %(prog)s                                    # Normal startup (session-mode 1)
  %(prog)s --session-mode -1                  # No session read/write (like old --no-session)
  %(prog)s --session-mode 0                   # Fresh start, save new session on exit
  %(prog)s --auto-exit 3                      # Exit automatically after 3 seconds (testing)
  %(prog)s --test-mode --session-mode -1      # Test mode without session state
  %(prog)s --auto-exit 2 --session-mode -1    # Quick startup test

Notes:
  --auto-exit is useful for automated testing to verify the application starts correctly
  --session-mode controls session behavior:
    -1: Don't load or save session (no read/write)
     0: Start fresh, save new session on exit (write only)
     1: Normal mode - load and save session (default)
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
        "--session-mode",
        type=int,
        default=1,
        choices=[-1, 0, 1],
        metavar="MODE",
        help="Session mode: -1=no read/write, 0=write only (fresh start), 1=normal (default)",
    )

    parser.add_argument(
        "--screenshot",
        type=str,
        metavar="PATH",
        help="Save screenshot to PATH after window is shown (e.g., /tmp/screenshot.png)",
    )

    parser.add_argument(
        "--print-session",
        action="store_true",
        help="Print the session file path and contents, then exit",
    )

    return parser.parse_args()


def main() -> int:
    """Run the Seismic Viewer application."""
    args = parse_arguments()

    # Handle --print-session flag
    if args.print_session:
        import json

        # Create a minimal UAS app to get access to SessionManager
        app = UASApplication("Seismic Viewer")

        # Get the session path by calling save() which returns the path
        session_path = SessionManager.get_instance().save()
        print(f"Session file path: {session_path}")

        try:
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            print("\nSession file contents:")
            print(json.dumps(session_data, indent=2))
            return 0
        except FileNotFoundError:
            print(f"Session file not found at: {session_path}")
            return 1
        except Exception as e:
            print(f"Error reading session file: {e}")
            return 1

    app = UASApplication("Seismic Viewer")

    # Determine session behavior from session_mode
    # -1: no read/write, 0: write only (fresh start), 1: normal (read/write)
    load_session = (args.session_mode == 1)  # Only load in mode 1
    save_session = (args.session_mode >= 0)  # Save in modes 0 and 1

    # Register global settings for session persistence (unless session_mode is -1)
    if args.session_mode >= 0:
        GlobalSettings.register()

    # Store screenshot path for main window (hack: use a class variable)
    if args.screenshot:
        SeismicMainWindow._pending_screenshot_path = args.screenshot

    # Store auto-exit seconds for main window (will be set up in showEvent)
    if args.auto_exit:
        SeismicMainWindow._pending_auto_exit_seconds = args.auto_exit

    return app.run(
        default_main_window_type="seismic_main",
        load_session=load_session,
        save_session=save_session
    )


if __name__ == "__main__":
    sys.exit(main())
