"""
Session management for UAS framework.

Handles save/load of application state to JSON files.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, Protocol

from PySide6.QtWidgets import QApplication

from .factory import FactoryRegistry

if TYPE_CHECKING:
    from .main_window import UASMainWindow


class SerializablePlugin(Protocol):
    """Protocol for objects that can be serialized/deserialized with sessions."""

    def serialize(self) -> dict[str, Any]:
        """Serialize the plugin state to a dictionary."""
        ...

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore the plugin state from a dictionary."""
        ...

def get_session_directory() -> Path:
    """Get the directory for storing session files (platform-specific location)."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".config"
    session_dir = base / "uas_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


class SessionManager:
    """Manages application session state and persistence.

    Singleton pattern - use get_instance() to access the global session manager.
    """

    _instance: SessionManager | None = None

    def __init__(self) -> None:
        self._main_windows: list[UASMainWindow] = []
        self._session_name: str = "default"
        self._session_file: Path | None = None
        self._auto_save_enabled: bool = True
        self._plugins: dict[str, SerializablePlugin] = {}

    @classmethod
    def get_instance(cls) -> SessionManager:
        """Get the global session manager instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the session manager (mainly for testing)."""
        cls._instance = None

    @property
    def main_windows(self) -> list[UASMainWindow]:
        """Get all main windows in the session."""
        return self._main_windows.copy()

    @property
    def session_name(self) -> str:
        """Get the current session name."""
        return self._session_name

    @property
    def auto_save_enabled(self) -> bool:
        """Check if auto-save on last window close is enabled."""
        return self._auto_save_enabled

    @auto_save_enabled.setter
    def auto_save_enabled(self, value: bool) -> None:
        """Enable or disable auto-save on last window close."""
        self._auto_save_enabled = value

    def add_main_window(self, window: UASMainWindow) -> None:
        """Add a main window to the session."""
        if window not in self._main_windows:
            self._main_windows.append(window)

    def remove_main_window(self, window: UASMainWindow) -> None:
        """Remove a main window from the session."""
        if window in self._main_windows:
            self._main_windows.remove(window)

    def register_plugin(self, name: str, plugin: SerializablePlugin) -> None:
        """Register a plugin for session serialization."""
        self._plugins[name] = plugin

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin from session serialization."""
        if name in self._plugins:
            del self._plugins[name]

    def create_main_window_from_state(self, state: dict[str, Any]) -> UASMainWindow:
        """Create a main window from serialized state using factory registry."""
        registry = FactoryRegistry.get_instance()
        cls = registry.get_main_window_class(state["type"])
        window = cls.create(state)
        self.add_main_window(window)
        window.show()
        return window

    def serialize(self) -> dict[str, Any]:
        """Serialize the entire session to a dict including all main windows and plugins."""
        session_dict = {
            "name": self._session_name,
            "timestamp": datetime.now().isoformat(),
            "main_windows": [w.serialize() for w in self._main_windows],
        }

        # Serialize all registered plugins
        session_dict["plugins"] = { name: plugin.serialize() for name, plugin in self._plugins.items() }
        return session_dict

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore the session from a dict by closing existing windows and creating new ones."""
        for window in self._main_windows[:]:
            window.close()
        self._main_windows.clear()

        if "name" in state:
            self._session_name = state["name"]

        # Deserialize all registered plugins
        for name, plugin_state in state.get("plugins", {}).items():
            if name in self._plugins:
                self._plugins[name].deserialize(plugin_state)

        if "main_windows" in state:
            for window_state in state["main_windows"]:
                self.create_main_window_from_state(window_state)

    def save(self, file_path: str | Path | None = None) -> Path:
        """Save the session to a JSON file, returns the path used."""
        if file_path is None:
            file_path = get_session_directory() / f"{self._session_name}.json"
        else:
            file_path = Path(file_path)

        self._session_file = file_path

        state = self.serialize()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        return file_path

    def load(self, file_path: str | Path) -> None:
        """Load a session from a JSON file and restore all windows."""
        file_path = Path(file_path)
        self._session_file = file_path

        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.deserialize(state)

    def new_session(self, name: str = "default") -> None:
        """Start a new empty session by closing all windows and clearing state."""
        for window in self._main_windows[:]:
            window.close()
        self._main_windows.clear()
        self._session_name = name
        self._session_file = None

    def get_recent_sessions(self, max_count: int = 10) -> list[Path]:
        """Get a list of recent session files sorted by modification time (newest first)."""
        session_dir = get_session_directory()
        sessions = list(session_dir.glob("*.json"))
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sessions[:max_count]

    def close_all(self, save: bool = True) -> None:
        """Close all main windows and optionally save the session first."""
        if save and self._main_windows:
            self.save()

        for window in self._main_windows[:]:
            window.close()
        self._main_windows.clear()


class UASApplication:
    """
    Helper class for running a UAS application.
    
    Purpose:
        Provides standard application lifecycle management including QApplication
        setup, session loading/saving, and graceful shutdown.
    
    Flow:
        1. Initializes QApplication with application name
        2. Attempts to load recent session or specified session file
        3. Falls back to creating default main window if no session found
        4. Sets up auto-save on quit
        5. Runs application event loop
    """

    def __init__(self, app_name: str = "UAS Application") -> None:
        self._app_name = app_name
        self._app: QApplication | None = None

    def run(
        self,
        default_main_window_type: str | None = None,
        session_file: str | Path | None = None,
        load_session: bool = True,
        save_session: bool = True,
    ) -> int:
        """Run the application, loading session or creating default window.

        Args:
            default_main_window_type: Type of main window to create if no session loaded
            session_file: Path to specific session file to load
            load_session: If False, skip loading any session (default: True)
            save_session: If False, skip saving session on exit (default: True)
        """

        self._app = QApplication(sys.argv)
        self._app.setApplicationName(self._app_name)

        session = SessionManager.get_instance()
        create_default_window = True
        if load_session:
            if session_file:
                try:
                    session.load(session_file)
                    create_default_window = False
                except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                    print(f"Failed to load session: {e}")
            else:
                recent = session.get_recent_sessions(1)
                if recent:
                    try:
                        session.load(recent[0])
                        create_default_window = False
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Failed to load recent session: {e}")

        if create_default_window:
            self._create_default_window(type_name=default_main_window_type)

        # Disable auto-save if save_session is False
        if not save_session:
            session.auto_save_enabled = False

        # Setup quit handler: save session only if save_session is True
        self._app.aboutToQuit.connect(lambda: session.close_all(save=save_session))

        return self._app.exec()


    def _create_default_window(self, type_name: str|None = None) -> None:
        if not type_name:
            return
        """Create a default main window using factory registry."""
        session = SessionManager.get_instance()
        registry = FactoryRegistry.get_instance()
        cls = registry.get_main_window_class(type_name)
        window = cls.create()
        session.add_main_window(window)
        window.show()
