"""
Geophysical Filter System for GeoSense.

Provides a modular filter framework with:
- Factory pattern for filter registration
- Hierarchical filter organization (main_type -> sub_type)
- Dynamic parameter specification
- Filter pipeline management
"""

from .base import BaseFilter, FilterParameterSpec, ParameterType
from .registry import FilterRegistry, register_filter
from .pipeline import FilterPipeline
from .filters_dialog import FiltersDialog

# Import filter modules to trigger registration
from . import frequency
from . import amplitude
from . import spatial

__all__ = [
    "BaseFilter",
    "FilterParameterSpec",
    "ParameterType",
    "FilterRegistry",
    "register_filter",
    "FilterPipeline",
    "FiltersDialog",
]
