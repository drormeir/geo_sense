"""
Demo application showcasing the UAS Framework.

Run with: python uas_demo.py
"""

from typing import Any
import sys
import time

from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QProgressBar,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
)

from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

from uas import (
    UASSubWindow,
    UASMainWindow,
    FactoryRegistry,
    UASApplication,
    UASVisualizationSubWindow,
    ThreadedSubWindowMixin,
    ObservableSubWindowMixin,
    ListenerSubWindowMixin,
    ShortcutMixin,
    install_shortcut_handler,
    SessionManager,
    auto_register,
)


@auto_register
class CounterSubWindow(ObservableSubWindowMixin, ShortcutMixin, UASSubWindow):
    """
    A counter subwindow that broadcasts its value changes.

    Purpose:
        Demonstrates observable subwindow functionality by maintaining a counter
        value that can be incremented/decremented and notifies listeners of changes.

    Flow:
        1. Initializes with counter value of 0
        2. Sets up UI with label and increment/decrement buttons
        3. Registers keyboard shortcuts for increment/decrement
        4. When value changes, updates display and notifies all listeners
        5. Supports serialization/deserialization to persist counter state
    """

    type_name = "counter"

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        """Initialize the counter subwindow with initial value of 0."""
        self._counter = 0
        super().__init__(main_window, parent)

    def on_create(self) -> None:
        """Set up the UI components including label, buttons, and shortcuts."""
        self.title = "Counter"
        layout = QVBoxLayout(self)

        self._label = QLabel("0")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("font-size: 48px; font-weight: bold;")
        layout.addWidget(self._label)

        increment_btn = QPushButton("Increment (+)")
        increment_btn.clicked.connect(self._increment)
        layout.addWidget(increment_btn)

        decrement_btn = QPushButton("Decrement (-)")
        decrement_btn.clicked.connect(self._decrement)
        layout.addWidget(decrement_btn)

        self.setMinimumSize(200, 200)

        self.register_shortcut("Ctrl++", self._increment, "Increment counter")
        self.register_shortcut("Ctrl+-", self._decrement, "Decrement counter")

    def _increment(self) -> None:
        """Increment the counter value and notify listeners."""
        self._counter += 1
        self._label.setText(str(self._counter))
        self.update_status(f"Counter: {self._counter}")
        self.notify_listeners("counter_value", self._counter)

    def _decrement(self) -> None:
        """Decrement the counter value and notify listeners."""
        self._counter -= 1
        self._label.setText(str(self._counter))
        self.update_status(f"Counter: {self._counter}")
        self.notify_listeners("counter_value", self._counter)

    def serialize(self) -> dict[str, Any]:
        """Serialize the counter state including current value."""
        state = super().serialize()
        state["counter"] = self._counter
        return state

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore the counter state from serialized data."""
        super().deserialize(state)
        if "counter" in state:
            self._counter = state["counter"]
            self._label.setText(str(self._counter))


@auto_register
class CounterListenerSubWindow(ListenerSubWindowMixin, UASSubWindow):
    """
    Listens to counter updates and displays history.

    Purpose:
        Demonstrates listener subwindow functionality by subscribing to counter
        value changes and displaying a history of all updates.

    Flow:
        1. Sets up UI with a list widget to display history
        2. Registers event handler for "counter_value" events
        3. Announces presence to find and subscribe to all counter subwindows
        4. When counter value changes, adds entry to history list
        5. Automatically scrolls to show latest entry
    """

    type_name = "counter_listener"

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        """Initialize the counter listener subwindow."""
        super().__init__(main_window, parent)

    def on_create(self) -> None:
        """Set up the UI and subscribe to counter value events."""
        self.title = "Counter History"
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Counter value history:"))

        self._list = QListWidget()
        layout.addWidget(self._list)

        self.setMinimumSize(200, 200)

        self.register_event_handler("counter_value", self._on_counter_update)
        self.announce_presence(["counter_value"])

    def _on_counter_update(self, event) -> None:
        """Handle counter value update events by adding to history list."""
        self._list.addItem(f"Value changed to: {event.data}")
        self._list.scrollToBottom()


@auto_register
class DrawingSubWindow(UASVisualizationSubWindow):
    """
    A visualization subwindow with drawing capabilities.

    Purpose:
        Demonstrates visualization subwindow functionality by displaying
        a canvas with pre-drawn geometric shapes.

    Flow:
        1. Initializes the base visualization components (scene and view)
        2. Sets up a 500x500 scene with various colored shapes
        3. Adds rectangles and ellipses to the scene
        4. Automatically zooms to fit all content in the view
    """

    type_name = "drawing"

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        """Initialize the drawing subwindow."""
        super().__init__(main_window, parent)

    def on_create(self) -> None:
        """Set up the graphics scene with sample shapes."""
        super().on_create()
        self.title = "Drawing Canvas"

        self.scene.setSceneRect(0, 0, 500, 500)

        rect = QGraphicsRectItem(50, 50, 100, 100)
        rect.setBrush(QBrush(QColor(100, 150, 200)))
        self.scene.addItem(rect)

        ellipse = QGraphicsEllipseItem(200, 100, 80, 80)
        ellipse.setBrush(QBrush(QColor(200, 100, 150)))
        self.scene.addItem(ellipse)

        rect2 = QGraphicsRectItem(150, 250, 150, 100)
        rect2.setBrush(QBrush(QColor(150, 200, 100)))
        self.scene.addItem(rect2)

        self.view.zoom_to_fit()


@auto_register
class ComputationSubWindow(ThreadedSubWindowMixin, UASSubWindow):
    """
    Demonstrates background thread computation.

    Purpose:
        Shows how to perform long-running computations in a background thread
        while keeping the UI responsive, with progress updates and cancellation support.

    Flow:
        1. Sets up UI with status label, progress bar, and control buttons
        2. When start is clicked, launches background thread with computation
        3. Worker thread simulates heavy work, emitting progress updates
        4. UI updates progress bar and status in real-time from thread signals
        5. Supports cancellation which gracefully stops the computation
        6. Displays result or error when computation completes
    """

    type_name = "computation"

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        """Initialize the computation subwindow."""
        super().__init__(main_window, parent)

    def on_create(self) -> None:
        """Set up the UI components for computation display and control."""
        self.title = "Background Computation"
        layout = QVBoxLayout(self)

        self._status_label = QLabel("Ready")
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._result_label = QLabel("")
        layout.addWidget(self._result_label)

        start_btn = QPushButton("Start Computation")
        start_btn.clicked.connect(self._start_computation)
        layout.addWidget(start_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel_computation)
        layout.addWidget(cancel_btn)

        self.setMinimumSize(250, 200)

    def _start_computation(self) -> None:
        """Start the background computation task."""
        if self.is_thread_running:
            self.update_status("Computation already running")
            return

        self._status_label.setText("Computing...")
        self._progress.setValue(0)
        self._result_label.setText("")

        self.start_background_work(
            self._heavy_computation,
            on_progress=self._on_progress,
            on_result=self._on_result,
            on_error=self._on_error,
            on_finished=self._on_finished,
        )

    def _heavy_computation(self, worker):
        """Simulate heavy computation in the background thread."""
        total = 0
        for i in range(100):
            if worker.is_cancelled():
                return None
            time.sleep(0.05)
            total += i
            worker.emit_progress(i + 1, f"Processing step {i + 1}")
        return total

    def _on_progress(self, percent: int, message: str) -> None:
        """Update progress bar and status label from worker thread."""
        self._progress.setValue(percent)
        self._status_label.setText(message)

    def _on_result(self, result) -> None:
        """Handle successful computation result."""
        if result is not None:
            self._result_label.setText(f"Result: {result}")

    def _on_error(self, error: str) -> None:
        """Handle computation errors."""
        self._result_label.setText(f"Error: {error}")

    def _on_finished(self) -> None:
        """Handle computation completion (success or cancellation)."""
        self._status_label.setText("Done" if not self.worker.is_cancelled() else "Cancelled")

    def _cancel_computation(self) -> None:
        """Cancel the running computation."""
        self.cancel_background_work()


@auto_register
class DemoMainWindow(UASMainWindow):
    """
    Demo main window with custom menus, toolbar, and dock.

    Purpose:
        Provides a complete demo application main window with menu items and
        toolbar buttons to create different types of subwindows.

    Flow:
        1. Sets up File menu with options to create counter, listener, drawing,
           and computation subwindows
        2. Adds session save and quit actions
        3. Creates toolbar with quick access buttons for subwindow creation
        4. Installs shortcut handler for context-aware keyboard shortcuts
        5. Handles subwindow creation through factory registry
    """

    type_name = "demo_main"

    def __init__(self, parent=None) -> None:
        """Initialize the demo main window with custom title."""
        super().__init__(parent)
        self.setWindowTitle("UAS Demo")

    def _setup_menus(self) -> None:
        """Set up the menu bar with File menu and subwindow creation actions."""
        super()._setup_menus()

        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        menubar.insertMenu(menubar.actions()[0], file_menu)

        new_counter = QAction("New &Counter", self)
        new_counter.setShortcut("Ctrl+1")
        new_counter.triggered.connect(lambda: self._new_subwindow("counter"))
        file_menu.addAction(new_counter)

        new_listener = QAction("New Counter &Listener", self)
        new_listener.setShortcut("Ctrl+2")
        new_listener.triggered.connect(lambda: self._new_subwindow("counter_listener"))
        file_menu.addAction(new_listener)

        new_drawing = QAction("New &Drawing", self)
        new_drawing.setShortcut("Ctrl+3")
        new_drawing.triggered.connect(lambda: self._new_subwindow("drawing"))
        file_menu.addAction(new_drawing)

        new_computation = QAction("New C&omputation", self)
        new_computation.setShortcut("Ctrl+4")
        new_computation.triggered.connect(lambda: self._new_subwindow("computation"))
        file_menu.addAction(new_computation)

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

        self._setup_toolbar()
        install_shortcut_handler(self)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar with buttons for creating subwindows."""
        toolbar = self.create_toolbar("Main Toolbar")

        toolbar.addAction("Counter", lambda: self._new_subwindow("counter"))
        toolbar.addAction("Listener", lambda: self._new_subwindow("counter_listener"))
        toolbar.addAction("Drawing", lambda: self._new_subwindow("drawing"))
        toolbar.addAction("Compute", lambda: self._new_subwindow("computation"))

    def _new_subwindow(self, type_name: str) -> None:
        """Create and add a new subwindow of the specified type."""
        cls = FactoryRegistry.get_instance().get_subwindow_class(type_name)
        subwindow = cls.create(self)
        self.add_subwindow(subwindow)

    def _save_session(self) -> None:
        """Save the current session to disk and show confirmation message."""

        path = SessionManager.get_instance().save()
        self._status_bar.showMessage(f"Session saved to {path}", 3000)

    def _quit_app(self) -> None:
        """Save session, disable auto-save, and quit the application."""

        session = SessionManager.get_instance()
        session.save()
        session.auto_save_enabled = False
        QApplication.quit()


def main() -> int:
    """Run the demo application."""
    app = UASApplication("UAS Demo")
    return app.run(default_main_window_type="demo_main")


if __name__ == "__main__":

    sys.exit(main())
