"""
Visualization subwindow with QGraphicsView/QGraphicsScene.

Provides built-in zoom, pan, and scroll capabilities for 2D visualizations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QVBoxLayout,
)
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QWheelEvent, QMouseEvent, QPainter

from .subwindow import UASSubWindow

if TYPE_CHECKING:
    from .main_window import UASMainWindow


class UASGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView with zoom and pan support.
    
    Purpose:
        Extends QGraphicsView with built-in zoom (mouse wheel) and pan (middle mouse)
        capabilities, zoom limits, and view state persistence for session management.
    
    Flow:
        1. Mouse wheel events trigger zoom in/out with configurable limits
        2. Middle mouse button drag enables panning
        3. Supports zoom to fit, zoom to rect, and reset view operations
        4. View state (zoom, scroll position, transform) can be serialized/deserialized
    """

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._panning = False
        self._pan_start = QPointF()

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    @property
    def zoom_factor(self) -> float:
        """Get the current zoom factor."""
        return self._zoom_factor

    def set_zoom_limits(self, min_zoom: float, max_zoom: float) -> None:
        """Set the minimum and maximum zoom limits."""
        self._min_zoom = min_zoom
        self._max_zoom = max_zoom

    def zoom_in(self, factor: float = 1.25) -> None:
        """Zoom in by the given factor (respects max zoom limit)."""
        new_zoom = self._zoom_factor * factor
        if new_zoom <= self._max_zoom:
            self._zoom_factor = new_zoom
            self.scale(factor, factor)

    def zoom_out(self, factor: float = 1.25) -> None:
        """Zoom out by the given factor (respects min zoom limit)."""
        new_zoom = self._zoom_factor / factor
        if new_zoom >= self._min_zoom:
            self._zoom_factor = new_zoom
            self.scale(1 / factor, 1 / factor)

    def set_zoom(self, zoom: float) -> None:
        """Set the zoom to an absolute value (clamped to zoom limits)."""
        zoom = max(self._min_zoom, min(self._max_zoom, zoom))
        scale_factor = zoom / self._zoom_factor
        self._zoom_factor = zoom
        self.scale(scale_factor, scale_factor)

    def zoom_to_fit(self) -> None:
        """Zoom to fit the entire scene in the view while maintaining aspect ratio."""
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()

    def zoom_to_rect(self, rect: QRectF) -> None:
        """Zoom to fit a specific rectangle in the view."""
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()

    def reset_view(self) -> None:
        """Reset zoom and pan to default (zoom factor 1.0, no transform)."""
        self.resetTransform()
        self._zoom_factor = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel events for zooming in/out."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events (middle button starts panning)."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events (updates panning if middle button is pressed)."""
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events (ends panning when middle button released)."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def get_view_state(self) -> dict[str, Any]:
        """Get the current view state (zoom, scroll position, transform) for serialization."""
        transform = self.transform()
        return {
            "zoom": self._zoom_factor,
            "h_scroll": self.horizontalScrollBar().value(),
            "v_scroll": self.verticalScrollBar().value(),
            "transform": {
                "m11": transform.m11(),
                "m12": transform.m12(),
                "m21": transform.m21(),
                "m22": transform.m22(),
                "dx": transform.dx(),
                "dy": transform.dy(),
            },
        }

    def set_view_state(self, state: dict[str, Any]) -> None:
        """Restore view state from serialization (zoom, scroll position, transform)."""
        if "zoom" in state:
            self._zoom_factor = state["zoom"]
        if "transform" in state:
            from PySide6.QtGui import QTransform

            t = state["transform"]
            transform = QTransform(
                t.get("m11", 1), t.get("m12", 0),
                t.get("m21", 0), t.get("m22", 1),
                t.get("dx", 0), t.get("dy", 0)
            )
            self.setTransform(transform)
        if "h_scroll" in state:
            self.horizontalScrollBar().setValue(state["h_scroll"])
        if "v_scroll" in state:
            self.verticalScrollBar().setValue(state["v_scroll"])


class UASVisualizationSubWindow(UASSubWindow):
    """
    Base class for visualization subwindows using QGraphicsView/QGraphicsScene.
    
    Purpose:
        Provides a foundation for 2D visualization subwindows with built-in
        graphics scene, view, zoom, pan, and scroll capabilities. Handles
        view state persistence for session management.
    
    Flow:
        1. Initializes with QGraphicsScene and UASGraphicsView
        2. Subclasses add graphics items to scene in on_create()
        3. View provides zoom (wheel) and pan (middle mouse) interactions
        4. Supports zoom to fit, zoom to selection, and reset view
        5. View state is serialized/deserialized for session persistence
    """

    def __init__(self, main_window: UASMainWindow, parent=None) -> None:
        self._scene: QGraphicsScene | None = None
        self._view: UASGraphicsView | None = None
        super().__init__(main_window, parent)

    @property
    def scene(self) -> QGraphicsScene:
        """Get the graphics scene."""
        return self._scene

    @property
    def view(self) -> UASGraphicsView:
        """Get the graphics view."""
        return self._view

    def on_create(self) -> None:
        """Set up the visualization components (scene and view)."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scene = QGraphicsScene(self)
        self._view = UASGraphicsView(self._scene, self)
        layout.addWidget(self._view)

        self.setMinimumSize(200, 200)

    def zoom_in(self) -> None:
        """Zoom in on the visualization and update status bar."""
        self._view.zoom_in()
        self.update_status(f"Zoom: {self._view.zoom_factor:.1%}")

    def zoom_out(self) -> None:
        """Zoom out on the visualization and update status bar."""
        self._view.zoom_out()
        self.update_status(f"Zoom: {self._view.zoom_factor:.1%}")

    def zoom_to_fit(self) -> None:
        """Zoom to fit the entire scene and update status bar."""
        self._view.zoom_to_fit()
        self.update_status("Zoom to fit")

    def zoom_to_selection(self, rect: QRectF) -> None:
        """Zoom to fit a selection rectangle and update status bar."""
        self._view.zoom_to_rect(rect)
        self.update_status("Zoom to selection")

    def reset_view(self) -> None:
        """Reset zoom and pan to defaults and update status bar."""
        self._view.reset_view()
        self.update_status("View reset")

    def on_mouse_hover(self, scene_pos: QPointF) -> None:
        """Called when mouse hovers over the scene (override to handle hover, default shows coordinates)."""
        self.update_status(f"Position: ({scene_pos.x():.1f}, {scene_pos.y():.1f})")

    def on_mouse_click(self, scene_pos: QPointF, button: Qt.MouseButton) -> None:
        """Called when mouse clicks on the scene (override to handle click events)."""
        pass

    def on_mouse_drag(self, start_pos: QPointF, current_pos: QPointF) -> None:
        """Called during mouse drag on the scene (override to handle drag events)."""
        pass

    def serialize(self) -> dict[str, Any]:
        """Serialize subwindow state including view state (zoom, scroll position)."""
        state = super().serialize()
        if self._view:
            state["view_state"] = self._view.get_view_state()
        return state

    def deserialize(self, state: dict[str, Any]) -> None:
        """Restore subwindow state including view state (zoom, scroll position)."""
        super().deserialize(state)
        if "view_state" in state and self._view:
            self._view.set_view_state(state["view_state"])
