"""
Base classes for the geophysical filter system.

Provides:
- ParameterType: Enum for parameter data types
- FilterParameterSpec: Dataclass defining filter parameters
- BaseFilter: Abstract base class for all filters
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
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

    @classmethod
    def describe(cls) -> str:
        """Return a description of the filter and all its parameters."""
        lines = [
            f"{cls.filter_name} ({cls.category})",
            f"  {cls.description}",
            "",
            "  Parameters:",
        ]
        for spec in cls.parameter_specs:
            # Build parameter info line
            param_line = f"    {spec.display_name} ({spec.name}): {spec.param_type.value}"
            if spec.units:
                param_line += f" [{spec.units}]"
            lines.append(param_line)

            # Add range info if applicable
            if spec.min_value is not None or spec.max_value is not None:
                range_parts = []
                if spec.min_value is not None:
                    range_parts.append(f"min={spec.min_value}")
                if spec.max_value is not None:
                    range_parts.append(f"max={spec.max_value}")
                lines.append(f"      Range: {', '.join(range_parts)}")

            # Add default value
            lines.append(f"      Default: {spec.default}")

            # Add tooltip/description if present
            if spec.tooltip:
                lines.append(f"      {spec.tooltip}")

        return "\n".join(lines)

    # Flag set by FiltersDialog to indicate demo is called from UI
    _demo_called_from_ui: bool = False

    @classmethod
    def render_demo_figure(
        cls,
        subplots: list[dict[str, Any]],
        figure_params: dict[str, Any] | None = None
    ) -> None:
        """
        Render a 2x2 demo figure from subplot specifications.

        Args:
            subplots: List of 4 dictionaries, one per subplot (top-left, top-right,
                     bottom-left, bottom-right). Each dict can contain:

                Common keys:
                    title: str - subplot title
                    xlabel: str - x-axis label
                    ylabel: str - y-axis label
                    grid: bool | float - show grid (True or alpha value)
                    legend: bool | dict - show legend (True or legend kwargs)
                    xlim: tuple - (min, max) for x-axis
                    ylim: tuple - (min, max) for y-axis
                    invert_xaxis: bool - invert x-axis
                    invert_yaxis: bool - invert y-axis
                    axvlines: list[dict] - vertical lines [{x, color, linestyle, alpha}, ...]
                    axhlines: list[dict] - horizontal lines [{y, color, linestyle, alpha}, ...]

                For line plots (type='plot' or default):
                    lines: list[dict] - line specifications, each with:
                        y: array - y data (required)
                        x: array - x data (optional, defaults to indices)
                        label: str - legend label
                        color: str - line color
                        linewidth: float - line width
                        alpha: float - transparency
                        linestyle: str - line style ('-', '--', ':', etc.)

                For image plots (type='imshow'):
                    data: 2D array - image data
                    extent: list - [x0, x1, y0, y1]
                    cmap: str - colormap name
                    vmin: float - color scale minimum
                    vmax: float - color scale maximum
                    colorbar: bool - show colorbar
                    aspect: str - aspect ratio ('auto', 'equal', etc.)

            figure_params: Dict with figure-level parameters:
                suptitle: str - figure title
                figsize: tuple - (width, height) in inches
        """
        import matplotlib.pyplot as plt

        fig_params = figure_params or {}
        figsize = fig_params.get('figsize', (12, 8))

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        if 'suptitle' in fig_params:
            fig.suptitle(fig_params['suptitle'])

        for ax, spec in zip(axes_flat, subplots):
            if not spec:  # Skip empty subplots
                ax.axis('off')
                continue

            plot_type = spec.get('type', 'plot')

            if plot_type == 'imshow':
                cls._render_imshow(ax, spec)
            else:
                cls._render_plot(ax, spec)

            # Common settings
            if 'title' in spec:
                ax.set_title(spec['title'])
            if 'xlabel' in spec:
                ax.set_xlabel(spec['xlabel'])
            if 'ylabel' in spec:
                ax.set_ylabel(spec['ylabel'])

            # Grid
            grid = spec.get('grid')
            if grid is True:
                ax.grid(True, alpha=0.3)
            elif isinstance(grid, (int, float)):
                ax.grid(True, alpha=grid)

            # Axis limits
            if 'xlim' in spec:
                ax.set_xlim(spec['xlim'])
            if 'ylim' in spec:
                ax.set_ylim(spec['ylim'])

            # Axis inversion
            if spec.get('invert_xaxis'):
                ax.invert_xaxis()
            if spec.get('invert_yaxis'):
                ax.invert_yaxis()

            # Vertical lines
            for vline in spec.get('axvlines', []):
                ax.axvline(
                    x=vline.get('x', 0),
                    color=vline.get('color', 'gray'),
                    linestyle=vline.get('linestyle', ':'),
                    alpha=vline.get('alpha', 0.7)
                )

            # Horizontal lines
            for hline in spec.get('axhlines', []):
                ax.axhline(
                    y=hline.get('y', 0),
                    color=hline.get('color', 'gray'),
                    linestyle=hline.get('linestyle', ':'),
                    alpha=hline.get('alpha', 0.7)
                )

            # Legend
            legend = spec.get('legend')
            if legend is True:
                ax.legend(fontsize=8)
            elif isinstance(legend, dict):
                ax.legend(**legend)

        plt.tight_layout()
        cls.show_demo_plot()

    @classmethod
    def _render_plot(cls, ax, spec: dict[str, Any]) -> None:
        """Render a line plot subplot."""
        lines = spec.get('lines', [])
        for line in lines:
            y = line.get('y')
            if y is None:
                continue
            x = line.get('x')
            kwargs = cls.get_args(line, ['label', 'color', 'linewidth', 'alpha', 'linestyle'])

            if x is not None:
                ax.plot(x, y, **kwargs)
            else:
                ax.plot(y, **kwargs)

    @classmethod
    def _render_imshow(cls, ax, spec: dict[str, Any]) -> None:
        """Render an imshow subplot."""
        import matplotlib.pyplot as plt

        data = spec.get('data')
        if data is None:
            return

        kwargs = cls.get_args(spec, ['extent', 'cmap', 'vmin', 'vmax', 'aspect'], [None, None, None, None, 'auto'])
        im = ax.imshow(data, **kwargs)

        if spec.get('colorbar'):
            plt.colorbar(im, ax=ax, shrink=spec.get('colorbar_shrink', 0.8))

    @staticmethod
    def get_args(source: dict[str, Any], keys: list[str], default_values: list[Any] = None) -> dict[str, Any]:
        """Get arguments from source."""
        target = {}
        if default_values is None:
            default_values = [None] * len(keys)
        for key, default_value in zip(keys, default_values):
            if key in source:
                target[key] = source[key]
            elif default_value is not None:
                target[key] = default_value
        return target

    @staticmethod
    def show_demo_plot() -> None:
        """
        Show matplotlib plot, handling Qt event loop properly.

        When running from command line, blocks until window is closed.
        When called from Qt application (via Demo button), does nothing
        as the caller handles the display.
        """
        import matplotlib.pyplot as plt

        # If called from UI, let the dialog handle plt.show()
        if BaseFilter._demo_called_from_ui:
            return

        # Command line - block until closed
        plt.show()

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the filter.

        Override in subclasses to provide an interactive demo with
        synthetic data and matplotlib visualization.

        Default implementation shows a message that no demo is available.
        """
        print(f"No demo available for {cls.filter_name}")

    @classmethod
    def has_demo(cls) -> bool:
        """Check if this filter has a custom demo implementation."""
        return cls.demo is not BaseFilter.demo

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
