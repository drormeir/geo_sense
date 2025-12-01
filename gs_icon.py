"""
GeoSense Icon Utility

Provides a shared function to create the "GS" icon for use across the application.
"""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt


def create_gs_icon() -> QIcon:
    """
    Create a custom icon with 'GS' text.

    Returns:
        QIcon with blue circular background and white "GS" text.
    """
    # Create a 32x32 pixmap
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)

    # Draw on the pixmap
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    # Draw background circle
    painter.setBrush(QColor(0, 120, 212))  # Blue background
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, 32, 32)

    # Draw "GS" text
    painter.setPen(QColor(255, 255, 255))  # White text
    font = QFont("Arial", 14, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "GS")

    painter.end()

    return QIcon(pixmap)
