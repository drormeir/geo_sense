"""
Filter pipeline for managing ordered filter chains.

Provides:
- FilterPipeline: Ordered list of filter instances with add/remove/reorder operations
"""

from __future__ import annotations
from typing import Any, Iterator

import numpy as np

from .base import BaseFilter
from .registry import FilterRegistry


class FilterPipeline:
    """
    Manages an ordered list of filter instances.

    Filters are applied sequentially in order.
    Supports add, remove, reorder operations.
    """

    def __init__(self) -> None:
        self._filters: list[BaseFilter] = []

    @property
    def filters(self) -> list[BaseFilter]:
        """Get the list of filters (copy)."""
        return list(self._filters)

    def __len__(self) -> int:
        return len(self._filters)

    def __iter__(self) -> Iterator[BaseFilter]:
        return iter(self._filters)

    def __getitem__(self, index: int) -> BaseFilter:
        return self._filters[index]

    def add_filter(self, filter_instance: BaseFilter) -> None:
        """Add a filter to the end of the pipeline."""
        self._filters.append(filter_instance)

    def insert_filter(self, index: int, filter_instance: BaseFilter) -> None:
        """Insert a filter at a specific position."""
        self._filters.insert(index, filter_instance)

    def remove_filter(self, index: int) -> BaseFilter:
        """Remove and return filter at index."""
        return self._filters.pop(index)

    def remove_by_id(self, instance_id: str) -> bool:
        """Remove filter by instance ID. Returns True if found."""
        for i, f in enumerate(self._filters):
            if f.instance_id == instance_id:
                self._filters.pop(i)
                return True
        return False

    def get_by_id(self, instance_id: str) -> BaseFilter | None:
        """Get filter by instance ID."""
        for f in self._filters:
            if f.instance_id == instance_id:
                return f
        return None

    def move_up(self, index: int) -> bool:
        """Move filter up one position. Returns True if moved."""
        if index <= 0 or index >= len(self._filters):
            return False
        self._filters[index], self._filters[index - 1] = \
            self._filters[index - 1], self._filters[index]
        return True

    def move_down(self, index: int) -> bool:
        """Move filter down one position. Returns True if moved."""
        if index < 0 or index >= len(self._filters) - 1:
            return False
        self._filters[index], self._filters[index + 1] = \
            self._filters[index + 1], self._filters[index]
        return True

    def clear(self) -> None:
        """Remove all filters."""
        self._filters.clear()

    def apply(self, data: np.ndarray|None, sample_interval: float) -> np.ndarray|None:
        """
        Apply all filters in sequence.

        Args:
            data: Input seismic data (nt x nx)
            sample_interval: Time between samples in seconds

        Returns:
            Filtered data
        """
        if data is None or len(self._filters) == 0:
            return data

        result = data.copy()  # Don't modify original
        for filter_instance in self._filters:
            result = filter_instance.apply(result, sample_interval)
        return result

    def serialize(self) -> list[dict[str, Any]]:
        """Serialize pipeline for persistence."""
        return [f.serialize() for f in self._filters]

    def deserialize(self, state: list[dict[str, Any]]) -> None:
        """Restore pipeline from serialized state."""
        self._filters.clear()
        registry = FilterRegistry.get_instance()
        for filter_state in state:
            filter_name = filter_state["filter_name"]
            filter_class = registry.get_filter_class(filter_name)
            filter_instance = filter_class.deserialize(filter_state)
            self._filters.append(filter_instance)
