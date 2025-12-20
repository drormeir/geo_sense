"""
Filters dialog for managing the filter pipeline.

Provides:
- FiltersDialog: Main dialog for filter selection, parameters, and pipeline management
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QGroupBox, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QLabel, QWidget,
)
from PySide6.QtCore import Qt

from .registry import FilterRegistry
from .base import BaseFilter, FilterParameterSpec, ParameterType
from .pipeline import FilterPipeline

if TYPE_CHECKING:
    from seismic_sub_win import SeismicSubWindow


class FiltersDialog(QDialog):
    """
    Dialog for managing filter pipeline.

    Layout:
    +-----------------------------------------------+
    | Filter Selection                              |
    | [Main Type v] [Sub Type v] [Add to Pipeline]  |
    +-----------------------------------------------+
    | Parameters (dynamic based on filter)          |
    | [QFormLayout with spinboxes/combos/checkboxes]|
    +-----------------------------------------------+
    | Filter Pipeline                               |
    | [QListWidget showing added filters]           |
    | [Move Up] [Move Down] [Remove] [Clear All]    |
    +-----------------------------------------------+
    | [Apply] [OK] [Cancel]                         |
    +-----------------------------------------------+
    """

    def __init__(self, parent: SeismicSubWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle("Filters")
        self.setMinimumWidth(500)
        self._parent = parent
        self._registry = FilterRegistry.get_instance()

        # Copy of pipeline for editing (don't modify original until OK/Apply)
        self._pipeline = FilterPipeline()
        if hasattr(parent, '_filter_pipeline'):
            self._pipeline.deserialize(parent._filter_pipeline.serialize())

        # Store original pipeline state for cancel
        self._original_pipeline_state = self._pipeline.serialize()

        # Track current filter being configured
        self._current_filter_class: type[BaseFilter] | None = None
        self._param_widgets: dict[str, QWidget] = {}

        self._create_ui()
        self._populate_categories()
        self._update_pipeline_list()

    def _create_ui(self) -> None:
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # === Filter Selection Group ===
        selection_group = QGroupBox("Filter Selection")
        selection_layout = QHBoxLayout()

        # Category combo
        selection_layout.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.setMinimumWidth(120)
        self._category_combo.currentTextChanged.connect(self._on_category_changed)
        selection_layout.addWidget(self._category_combo)

        # Filter name combo
        selection_layout.addWidget(QLabel("Filter:"))
        self._filter_name_combo = QComboBox()
        self._filter_name_combo.setMinimumWidth(120)
        self._filter_name_combo.currentTextChanged.connect(self._on_filter_name_changed)
        selection_layout.addWidget(self._filter_name_combo)

        selection_layout.addStretch()

        # Add button
        self._add_button = QPushButton("Add to Pipeline")
        self._add_button.clicked.connect(self._on_add_filter)
        selection_layout.addWidget(self._add_button)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        # === Parameters Group ===
        self._params_group = QGroupBox("Parameters")
        self._params_layout = QFormLayout()
        self._params_group.setLayout(self._params_layout)
        layout.addWidget(self._params_group)

        # === Pipeline Group ===
        pipeline_group = QGroupBox("Filter Pipeline")
        pipeline_layout = QVBoxLayout()

        # Pipeline list
        self._pipeline_list = QListWidget()
        self._pipeline_list.setMinimumHeight(150)
        self._pipeline_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._pipeline_list.itemSelectionChanged.connect(self._on_pipeline_selection_changed)
        self._pipeline_list.itemDoubleClicked.connect(self._on_pipeline_item_double_clicked)
        pipeline_layout.addWidget(self._pipeline_list)

        # Pipeline control buttons
        buttons_layout = QHBoxLayout()

        self._move_up_btn = QPushButton("Move Up")
        self._move_up_btn.clicked.connect(self._on_move_up)
        buttons_layout.addWidget(self._move_up_btn)

        self._move_down_btn = QPushButton("Move Down")
        self._move_down_btn.clicked.connect(self._on_move_down)
        buttons_layout.addWidget(self._move_down_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_filter)
        buttons_layout.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._on_clear_all)
        buttons_layout.addWidget(self._clear_btn)

        pipeline_layout.addLayout(buttons_layout)
        pipeline_group.setLayout(pipeline_layout)
        layout.addWidget(pipeline_group)

        # === Dialog Buttons ===
        button_layout = QHBoxLayout()

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self._apply_btn)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self._on_cancel)
        button_layout.addWidget(button_box)

        layout.addLayout(button_layout)

        self._update_button_states()

    def _populate_categories(self) -> None:
        """Fill category combo with registered categories."""
        self._category_combo.clear()
        categories = self._registry.get_categories()
        self._category_combo.addItems(categories)
        if categories:
            self._on_category_changed(categories[0])

    def _on_category_changed(self, category: str) -> None:
        """Update filter name combo when category changes."""
        self._filter_name_combo.blockSignals(True)
        self._filter_name_combo.clear()
        filter_names = self._registry.get_filter_names(category)
        self._filter_name_combo.addItems(filter_names)
        self._filter_name_combo.blockSignals(False)
        if filter_names:
            self._on_filter_name_changed(filter_names[0])

    def _on_filter_name_changed(self, filter_name: str) -> None:
        """Update parameter widgets when filter selection changes."""
        if not filter_name:
            self._current_filter_class = None
            self._clear_param_widgets()
            return

        try:
            self._current_filter_class = self._registry.get_filter_class(filter_name)
            self._rebuild_param_widgets()
        except KeyError:
            self._current_filter_class = None
            self._clear_param_widgets()

    def _clear_param_widgets(self) -> None:
        """Clear all parameter widgets."""
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets.clear()

    def _rebuild_param_widgets(self) -> None:
        """Rebuild parameter widgets for current filter class."""
        self._clear_param_widgets()

        if self._current_filter_class is None:
            return

        # Create widget for each parameter
        for spec in self._current_filter_class.parameter_specs:
            widget = self._create_param_widget(spec)
            label = spec.display_name
            if spec.units:
                label += f" [{spec.units}]"
            self._params_layout.addRow(label + ":", widget)
            self._param_widgets[spec.name] = widget

    def _create_param_widget(self, spec: FilterParameterSpec) -> QWidget:
        """Create appropriate widget for a parameter spec."""
        if spec.param_type == ParameterType.INT:
            widget = QSpinBox()
            widget.setRange(
                int(spec.min_value) if spec.min_value is not None else -999999,
                int(spec.max_value) if spec.max_value is not None else 999999
            )
            if spec.step:
                widget.setSingleStep(int(spec.step))
            widget.setValue(int(spec.default))
            widget.setToolTip(spec.tooltip)
            return widget

        elif spec.param_type == ParameterType.FLOAT:
            widget = QDoubleSpinBox()
            widget.setRange(
                float(spec.min_value) if spec.min_value is not None else -999999.0,
                float(spec.max_value) if spec.max_value is not None else 999999.0
            )
            widget.setDecimals(spec.decimals)
            if spec.step:
                widget.setSingleStep(float(spec.step))
            widget.setValue(float(spec.default))
            widget.setToolTip(spec.tooltip)
            return widget

        elif spec.param_type == ParameterType.CHOICE:
            widget = QComboBox()
            widget.addItems(spec.choices or [])
            widget.setCurrentText(str(spec.default))
            widget.setToolTip(spec.tooltip)
            return widget

        elif spec.param_type == ParameterType.BOOL:
            widget = QCheckBox()
            widget.setChecked(bool(spec.default))
            widget.setToolTip(spec.tooltip)
            return widget

        else:  # STRING
            widget = QLineEdit()
            widget.setText(str(spec.default))
            widget.setToolTip(spec.tooltip)
            return widget

    def _get_params_from_widgets(self) -> dict[str, Any]:
        """Extract current parameter values from widgets."""
        params: dict[str, Any] = {}
        if self._current_filter_class is None:
            return params

        for spec in self._current_filter_class.parameter_specs:
            widget = self._param_widgets.get(spec.name)
            if widget is None:
                continue

            if spec.param_type == ParameterType.INT:
                params[spec.name] = widget.value()
            elif spec.param_type == ParameterType.FLOAT:
                params[spec.name] = widget.value()
            elif spec.param_type == ParameterType.CHOICE:
                params[spec.name] = widget.currentText()
            elif spec.param_type == ParameterType.BOOL:
                params[spec.name] = widget.isChecked()
            else:  # STRING
                params[spec.name] = widget.text()

        return params

    def _on_add_filter(self) -> None:
        """Add current filter configuration to pipeline."""
        filter_name = self._filter_name_combo.currentText()
        if not filter_name:
            return

        params = self._get_params_from_widgets()

        # Create new filter instance with current params
        new_filter = self._registry.create_filter(filter_name, **params)
        self._pipeline.add_filter(new_filter)
        self._update_pipeline_list()

    def _update_pipeline_list(self) -> None:
        """Refresh the pipeline list widget."""
        self._pipeline_list.clear()
        for i, f in enumerate(self._pipeline.filters):
            # Create display text showing filter type and key params
            text = f"{i+1}. {f.category} - {f.filter_name}"
            param_summary = self._get_param_summary(f)
            if param_summary:
                text += f" ({param_summary})"

            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, f.instance_id)
            self._pipeline_list.addItem(item)

        self._update_button_states()

    def _get_param_summary(self, filter_instance: BaseFilter) -> str:
        """Get brief parameter summary for list display."""
        parts = []
        specs = filter_instance.parameter_specs[:2]  # Show first 2 params
        for spec in specs:
            value = filter_instance.get_parameter(spec.name)
            if spec.units:
                parts.append(f"{value}{spec.units}")
            else:
                parts.append(f"{spec.display_name}={value}")
        return ", ".join(parts)

    def _on_pipeline_selection_changed(self) -> None:
        """Handle pipeline list selection change."""
        self._update_button_states()

    def _on_pipeline_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Load filter params into editor when double-clicked."""
        instance_id = item.data(Qt.ItemDataRole.UserRole)
        filter_instance = self._pipeline.get_by_id(instance_id)
        if filter_instance is None:
            return

        # Set combos to match filter type
        self._category_combo.setCurrentText(filter_instance.category)
        self._filter_name_combo.setCurrentText(filter_instance.filter_name)

        # Load params into widgets
        self._load_params_to_widgets(filter_instance.parameters)

    def _load_params_to_widgets(self, params: dict[str, Any]) -> None:
        """Load parameter values into widgets."""
        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue

            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on current state."""
        has_selection = len(self._pipeline_list.selectedItems()) > 0
        has_filters = len(self._pipeline) > 0
        has_filter_types = self._category_combo.count() > 0

        self._add_button.setEnabled(has_filter_types)
        self._move_up_btn.setEnabled(has_selection)
        self._move_down_btn.setEnabled(has_selection)
        self._remove_btn.setEnabled(has_selection)
        self._clear_btn.setEnabled(has_filters)

    def _get_selected_index(self) -> int:
        """Get currently selected pipeline index, or -1."""
        items = self._pipeline_list.selectedItems()
        if not items:
            return -1
        return self._pipeline_list.row(items[0])

    def _on_move_up(self) -> None:
        """Move selected filter up in pipeline."""
        index = self._get_selected_index()
        if self._pipeline.move_up(index):
            self._update_pipeline_list()
            self._pipeline_list.setCurrentRow(index - 1)

    def _on_move_down(self) -> None:
        """Move selected filter down in pipeline."""
        index = self._get_selected_index()
        if self._pipeline.move_down(index):
            self._update_pipeline_list()
            self._pipeline_list.setCurrentRow(index + 1)

    def _on_remove_filter(self) -> None:
        """Remove selected filter from pipeline."""
        index = self._get_selected_index()
        if index >= 0:
            self._pipeline.remove_filter(index)
            self._update_pipeline_list()

    def _on_clear_all(self) -> None:
        """Clear all filters from pipeline."""
        self._pipeline.clear()
        self._update_pipeline_list()

    def _apply_to_parent(self) -> None:
        """Apply current pipeline to parent's data."""
        if hasattr(self._parent, '_filter_pipeline'):
            self._parent._filter_pipeline.deserialize(self._pipeline.serialize())
        if hasattr(self._parent, '_apply_filters_and_render'):
            self._parent._apply_filters_and_render()

    def _on_apply(self) -> None:
        """Apply current pipeline to seismic data (live preview)."""
        self._apply_to_parent()

    def _on_ok(self) -> None:
        """Accept changes and close dialog."""
        self._apply_to_parent()
        self.accept()

    def _on_cancel(self) -> None:
        """Cancel changes and close dialog."""
        # Restore original pipeline
        if hasattr(self._parent, '_filter_pipeline'):
            self._parent._filter_pipeline.deserialize(self._original_pipeline_state)
        if hasattr(self._parent, '_apply_filters_and_render'):
            self._parent._apply_filters_and_render()
        self.reject()
