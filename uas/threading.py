"""
Background thread support for subwindows.

Provides a managed background thread that can send events to the main thread.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal, QObject


class ThreadStatus(Enum):
    """
    Status of a background thread.
    
    Purpose:
        Enumeration of possible states for background worker threads.
    """

    IDLE = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    ERROR = auto()


class WorkerSignals(QObject):
    """
    Signals for communicating from worker thread to main thread.
    
    Purpose:
        Qt signals that allow the background worker thread to communicate
        with the main GUI thread safely. All signals are automatically
        queued to the main thread's event loop.
    """

    started = Signal()
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()
    custom_event = Signal(str, object)


class BackgroundWorker(QThread):
    """
    Background worker thread for heavy computations.
    
    Purpose:
        QThread subclass that executes work functions in a background thread,
        allowing long-running computations without blocking the GUI. Supports
        progress updates, cancellation, and error handling.
    
    Flow:
        1. Work function is set via set_work() with optional args/kwargs
        2. Thread started via start(), which calls run()
        3. run() executes work function, passing worker instance as first arg
        4. Work function can call emit_progress(), emit_event(), and check is_cancelled()
        5. Signals are emitted for progress, result, error, and finished
        6. Main thread receives signals and updates UI accordingly
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.signals = WorkerSignals()
        self._work_func: Callable[..., Any] | None = None
        self._args: tuple = ()
        self._kwargs: dict = {}
        self._cancelled = False
        self._status = ThreadStatus.IDLE

    @property
    def status(self) -> ThreadStatus:
        """Get the current thread status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if the thread is currently running."""
        return self._status == ThreadStatus.RUNNING

    def set_work(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> None:
        """Set the work function to execute with optional arguments."""
        self._work_func = func
        self._args = args
        self._kwargs = kwargs

    def cancel(self) -> None:
        """Request cancellation of the running work (work function should check is_cancelled())."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested (work functions should call this periodically)."""
        return self._cancelled

    def emit_progress(self, percent: int, message: str = "") -> None:
        """Emit a progress update signal to the main thread."""
        self.signals.progress.emit(percent, message)

    def emit_event(self, event_name: str, data: Any = None) -> None:
        """Emit a custom event signal to the main thread."""
        self.signals.custom_event.emit(event_name, data)

    def run(self) -> None:
        """Execute the work function in the background thread."""
        if self._work_func is None:
            return

        self._cancelled = False
        self._status = ThreadStatus.RUNNING
        self.signals.started.emit()

        try:
            result = self._work_func(self, *self._args, **self._kwargs)
            if self._cancelled:
                self._status = ThreadStatus.CANCELLED
            else:
                self._status = ThreadStatus.FINISHED
                self.signals.result.emit(result)
        except Exception as e:
            self._status = ThreadStatus.ERROR
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ThreadedSubWindowMixin:
    """
    Mixin class that adds background thread support to a subwindow.
    
    Purpose:
        Provides easy integration of background threading for subwindows that
        need to perform long-running computations. Handles worker lifecycle,
        progress callbacks, and serialization safety.
    
    Flow:
        1. Subwindow calls start_background_work() with work function and callbacks
        2. Mixin creates BackgroundWorker and connects signals to callbacks
        3. Worker executes in background thread, emitting progress/result/error signals
        4. Callbacks update UI on main thread
        5. Supports cancellation via cancel_background_work()
        6. Can block serialization while thread is running (configurable)
    """

    def __init__(self, *args, **kwargs) -> None:
        self._worker: BackgroundWorker | None = None
        self._allow_serialize_while_running = False
        super().__init__(*args, **kwargs)

    @property
    def worker(self) -> BackgroundWorker | None:
        """Get the current background worker."""
        return self._worker

    @property
    def is_thread_running(self) -> bool:
        """Check if a background thread is running."""
        return self._worker is not None and self._worker.is_running

    @property
    def thread_status(self) -> ThreadStatus:
        """Get the current thread status."""
        if self._worker is None:
            return ThreadStatus.IDLE
        return self._worker.status

    def allow_serialize_while_running(self, allow: bool) -> None:
        """Set whether serialization is allowed while thread is running."""
        self._allow_serialize_while_running = allow

    def can_serialize(self) -> bool:
        """Check if the subwindow can be serialized (returns False if thread running and blocked)."""
        if self.is_thread_running and not self._allow_serialize_while_running:
            return False
        return True

    def start_background_work(
        self,
        func: Callable[..., Any],
        *args,
        on_progress: Callable[[int, str], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_finished: Callable[[], None] | None = None,
        on_event: Callable[[str, Any], None] | None = None,
        **kwargs
    ) -> BackgroundWorker:
        """Start a background work task with optional progress/result/error callbacks."""
        if self._worker is not None and self._worker.isRunning():
            raise RuntimeError("A background task is already running")

        self._worker = BackgroundWorker(self)
        self._worker.set_work(func, *args, **kwargs)

        if on_progress:
            self._worker.signals.progress.connect(on_progress)
        if on_result:
            self._worker.signals.result.connect(on_result)
        if on_error:
            self._worker.signals.error.connect(on_error)
        if on_finished:
            self._worker.signals.finished.connect(on_finished)
        if on_event:
            self._worker.signals.custom_event.connect(on_event)

        self._worker.start()
        return self._worker

    def cancel_background_work(self) -> None:
        """Request cancellation of the current background work task."""
        if self._worker is not None:
            self._worker.cancel()

    def wait_for_thread(self, timeout_ms: int = -1) -> bool:
        """Wait for the background thread to finish (returns True if finished, False if timeout)."""
        if self._worker is None:
            return True
        if timeout_ms < 0:
            self._worker.wait()
            return True
        return self._worker.wait(timeout_ms)
