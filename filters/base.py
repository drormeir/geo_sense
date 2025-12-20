"""
Base classes for the geophysical filter system.

Provides:
- ParameterType: Enum for parameter data types
- FilterParameterSpec: Dataclass defining filter parameters
- BaseFilter: Abstract base class for all filters
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
import uuid

import numpy as np


class ParameterType(Enum):
    """Types of filter parameters."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    CHOICE = "choice"
    BOOL = "bool"


@dataclass
class FilterParameterSpec:
    """
    Specification for a single filter parameter.

    Defines the metadata needed to create appropriate UI widgets
    and validate parameter values.
    """
    name: str                                    # Internal parameter name
    display_name: str                            # User-facing label
    param_type: ParameterType                    # Data type
    default: Any                                 # Default value
    min_value: float | int | None = None         # Min for numeric types
    max_value: float | int | None = None         # Max for numeric types
    choices: list[str] | None = None             # Options for CHOICE type
    step: float | int | None = None              # Step size for spinboxes
    decimals: int = 3                            # Decimal places for floats
    tooltip: str = ""                            # Help text
    units: str = ""                              # Units label (Hz, ms, etc.)

    def validate(self, value: Any) -> tuple[bool, str]:
        """
        Validate a value against this spec.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.param_type == ParameterType.INT:
            if not isinstance(value, (int, float)):
                return False, f"{self.display_name} must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.display_name} must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.display_name} must be <= {self.max_value}"

        elif self.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"{self.display_name} must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.display_name} must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.display_name} must be <= {self.max_value}"

        elif self.param_type == ParameterType.CHOICE:
            if self.choices and str(value) not in self.choices:
                return False, f"{self.display_name} must be one of {self.choices}"

        elif self.param_type == ParameterType.BOOL:
            if not isinstance(value, bool):
                return False, f"{self.display_name} must be a boolean"

        return True, ""

    def coerce(self, value: Any) -> Any:
        """Coerce value to correct type, clamping to range if needed."""
        if self.param_type == ParameterType.INT:
            value = int(value)
            if self.min_value is not None:
                value = max(int(self.min_value), value)
            if self.max_value is not None:
                value = min(int(self.max_value), value)
            return value

        elif self.param_type == ParameterType.FLOAT:
            value = float(value)
            if self.min_value is not None:
                value = max(float(self.min_value), value)
            if self.max_value is not None:
                value = min(float(self.max_value), value)
            return value

        elif self.param_type == ParameterType.BOOL:
            return bool(value)

        return value


class BaseFilter(ABC):
    """
    Abstract base class for all geophysical filters.

    Each filter class defines:
    - category: Category for UI grouping (Frequency, Spatial, Amplitude, etc.)
    - filter_name: Unique filter name (Bandpass, AGC, etc.)
    - parameter_specs: List of FilterParameterSpec defining parameters
    - apply(): The actual filtering logic
    """

    # Class attributes (set by subclasses)
    category: ClassVar[str]
    filter_name: ClassVar[str]  # Must be unique across all filters
    description: ClassVar[str] = ""
    parameter_specs: ClassVar[list[FilterParameterSpec]] = []

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize filter with parameters.

        Args:
            **kwargs: Parameter values (names must match parameter_specs)
        """
        self._id = str(uuid.uuid4())
        self._parameters: dict[str, Any] = {}

        # Initialize with defaults
        for spec in self.parameter_specs:
            self._parameters[spec.name] = spec.default

        # Override with provided values
        for name, value in kwargs.items():
            if name in self._parameters:
                # Find spec and coerce value
                for spec in self.parameter_specs:
                    if spec.name == name:
                        self._parameters[name] = spec.coerce(value)
                        break

    @property
    def instance_id(self) -> str:
        """Unique identifier for this filter instance."""
        return self._id

    @property
    def parameters(self) -> dict[str, Any]:
        """Current parameter values (copy)."""
        return dict(self._parameters)

    def get_parameter(self, name: str) -> Any:
        """Get a parameter value by name."""
        return self._parameters.get(name)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter value."""
        if name in self._parameters:
            # Find spec and coerce value
            for spec in self.parameter_specs:
                if spec.name == name:
                    self._parameters[name] = spec.coerce(value)
                    break

    @classmethod
    def get_display_name(cls) -> str:
        """Return display name for UI."""
        return f"{cls.category} - {cls.filter_name}"

    @abstractmethod
    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """
        Apply the filter to seismic data.

        Args:
            data: 2D numpy array (nt x nx) - time samples x traces
            sample_interval: Time interval between samples in seconds

        Returns:
            Filtered data array of same shape
        """
        pass

    def serialize(self) -> dict[str, Any]:
        """Serialize filter state for persistence."""
        return {
            "filter_name": self.filter_name,
            "instance_id": self._id,
            "parameters": dict(self._parameters),
        }

    @classmethod
    def deserialize(cls, state: dict[str, Any]) -> BaseFilter:
        """Create filter instance from serialized state."""
        instance = cls(**state.get("parameters", {}))
        if "instance_id" in state:
            instance._id = state["instance_id"]
        return instance
