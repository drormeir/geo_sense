import sys
import numpy as np
import segyio
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QFileDialog, QMessageBox)
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SegyCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for displaying SEGY data"""
    def __init__(self, parent=None):
        fig = Figure(figsize=(10, 6))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
    def plot_segy(self, filename):
        """Load and display a SEGY file"""
        self.axes.clear()
        try:
            with segyio.open(filename, 'r') as f:
                data = f.trace.raw[:].T
                self.axes.imshow(data, aspect='auto', cmap='seismic',
                               vmin=-np.percentile(np.abs(data), 95),
                               vmax=np.percentile(np.abs(data), 95))
                self.axes.set_xlabel('Trace Number')
                self.axes.set_ylabel('Sample Number')
                self.axes.set_title(filename.split('/')[-1])
            self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")


class SegyViewerWindow(QMainWindow):
    """Main window with tabs for multiple SEGY files"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEGY Viewer")
        self.resize(1000, 700)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)
        
        # Create menus
        self.create_menus()
        
    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open SEGY...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        new_window_action = QAction("&New Window", self)
        new_window_action.setShortcut("Ctrl+N")
        new_window_action.triggered.connect(self.new_window)
        file_menu.addAction(new_window_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        # Add edit actions later
        
        # View menu
        view_menu = menubar.addMenu("&View")
        # Add view actions later
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        agc_action = QAction("Apply &AGC", self)
        agc_action.triggered.connect(self.apply_agc)
        tools_menu.addAction(agc_action)
        
    def open_file(self):
        """Open SEGY file dialog"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open SEGY File", "", 
            "SEGY Files (*.sgy *.segy);;All Files (*)"
        )
        
        if filename:
            # Create new canvas
            canvas = SegyCanvas()
            canvas.plot_segy(filename)
            
            # Add as new tab
            tab_name = filename.split('/')[-1]
            self.tabs.addTab(canvas, tab_name)
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
    
    def close_tab(self, index):
        """Close a tab"""
        self.tabs.removeTab(index)
    
    def new_window(self):
        """Create a new viewer window"""
        new_win = SegyViewerWindow()
        new_win.show()
        windows.append(new_win)  # Keep reference
    
    def apply_agc(self):
        """Apply AGC to current tab (placeholder)"""
        QMessageBox.information(self, "AGC", "AGC function not yet implemented")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Keep track of windows
    windows = []
    
    # Create first window
    main_window = SegyViewerWindow()
    main_window.show()
    windows.append(main_window)
    
    sys.exit(app.exec())
