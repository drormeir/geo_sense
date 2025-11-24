"""
Observer/Listener pattern for communication between subwindows.

Allows subwindows to subscribe to data updates from other subwindows
and receive notifications when data changes.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from weakref import WeakSet, ref

if TYPE_CHECKING:
    from .subwindow import UASSubWindow


class DataEvent:
    """
    Represents a data change event.
    
    Purpose:
        Encapsulates information about a data change event, including the source
        subwindow, the data key that changed, optional data payload, and event type.
    """

    def __init__(
        self,
        source: UASSubWindow,
        key: str,
        data: Any = None,
        event_type: str = "update"
    ) -> None:
        """Create a data event with source, key, optional data, and event type."""
        self.source = source
        self.key = key
        self.data = data
        self.event_type = event_type


class ObservableSubWindowMixin:
    """
    Mixin that makes a subwindow observable (can be listened to).
    
    Purpose:
        Provides observer pattern functionality allowing subwindows to broadcast
        data changes to registered listeners. Uses weak references to prevent
        memory leaks.
    
    Flow:
        1. Listeners register interest in specific data keys via add_listener()
        2. When data changes, observable calls notify_listeners() with key and data
        3. All registered listeners for that key receive DataEvent notifications
        4. Listeners can be removed individually or all at once
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the observable mixin with empty listener dictionary."""
        self._listeners: dict[str, WeakSet[ListenerSubWindowMixin]] = {}
        super().__init__(*args, **kwargs)

    def add_listener(self, key: str, listener: ListenerSubWindowMixin) -> None:
        """Add a listener for a specific data key."""
        if key not in self._listeners:
            self._listeners[key] = WeakSet()
        self._listeners[key].add(listener)

    def remove_listener(self, key: str, listener: ListenerSubWindowMixin) -> None:
        """Remove a listener for a specific data key."""
        if key in self._listeners:
            self._listeners[key].discard(listener)

    def remove_all_listeners(self, key: str | None = None) -> None:
        """Remove all listeners, optionally for a specific key."""
        if key is None:
            self._listeners.clear()
        elif key in self._listeners:
            self._listeners[key].clear()

    def notify_listeners(
        self,
        key: str,
        data: Any = None,
        event_type: str = "update"
    ) -> None:
        """Notify all listeners of a data change by creating and sending DataEvent."""
        if key not in self._listeners:
            return

        event = DataEvent(self, key, data, event_type)
        for listener in list(self._listeners[key]):
            try:
                listener.on_data_event(event)
            except Exception:
                pass

    def get_listener_count(self, key: str) -> int:
        """Get the number of listeners for a key."""
        if key not in self._listeners:
            return 0
        return len(self._listeners[key])

    def get_listened_keys(self) -> list[str]:
        """Get all keys that have active listeners."""
        return [k for k, v in self._listeners.items() if len(v) > 0]


class ListenerSubWindowMixin:
    """
    Mixin that allows a subwindow to listen to other subwindows.
    
    Purpose:
        Provides listener functionality allowing subwindows to subscribe to
        data changes from observable subwindows. Supports automatic discovery
        of observable sources and event handler registration.
    
    Flow:
        1. Listener registers event handlers for specific data keys
        2. Listener subscribes to observable sources (manually or via find_and_subscribe)
        3. When observable notifies listeners, listener receives DataEvent
        4. Event is routed to registered handler or on_data_event() method
        5. On close, all subscriptions are automatically cleaned up
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the listener mixin with empty subscription and handler dictionaries."""
        self._subscriptions: dict[str, list[ref]] = {}
        self._event_handlers: dict[str, Callable[[DataEvent], None]] = {}
        super().__init__(*args, **kwargs)

    def subscribe_to(self, source: ObservableSubWindowMixin, key: str) -> None:
        """Subscribe to updates from a specific observable source for a data key."""
        source.add_listener(key, self)
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(ref(source))

    def unsubscribe_from(self, source: ObservableSubWindowMixin, key: str) -> None:
        """Unsubscribe from a specific observable source for a data key."""
        source.remove_listener(key, self)
        if key in self._subscriptions:
            self._subscriptions[key] = [
                r for r in self._subscriptions[key] if r() is not None and r() != source
            ]

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all sources and clear subscriptions."""
        for key, sources in self._subscriptions.items():
            for source_ref in sources:
                source = source_ref()
                if source is not None:
                    source.remove_listener(key, self)
        self._subscriptions.clear()

    def register_event_handler(
        self,
        key: str,
        handler: Callable[[DataEvent], None]
    ) -> None:
        """Register a handler function for a specific data key."""
        self._event_handlers[key] = handler

    def on_data_event(self, event: DataEvent) -> None:
        """Called when a subscribed data source sends an update (override or use register_event_handler)."""
        if event.key in self._event_handlers:
            self._event_handlers[event.key](event)

    def find_observable_subwindows(
        self,
        key: str | None = None
    ) -> list[ObservableSubWindowMixin]:
        """Find all observable subwindows in the session, optionally filtered by key."""
        from .session import SessionManager

        results = []
        session = SessionManager.get_instance()

        for main_window in session.main_windows:
            for subwindow in main_window.subwindows:
                if isinstance(subwindow, ObservableSubWindowMixin):
                    if key is None or key in subwindow._listeners:
                        results.append(subwindow)

        return results

    def find_and_subscribe(self, key: str) -> int:
        """Find all observable subwindows and subscribe to them for a key, returns count."""
        sources = self.find_observable_subwindows()
        count = 0
        for source in sources:
            if source is not self:
                self.subscribe_to(source, key)
                count += 1
        return count

    def announce_presence(self, interested_keys: list[str]) -> None:
        """Announce this listener's presence and subscribe to all matching observables."""
        for key in interested_keys:
            self.find_and_subscribe(key)

    def on_close(self) -> None:
        """Clean up subscriptions when closing by unsubscribing from all sources."""
        self.unsubscribe_all()
        if hasattr(super(), 'on_close'):
            super().on_close()
