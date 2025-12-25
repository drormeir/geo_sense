"""
Filter registry for the geophysical filter system.

Provides:
- FilterRegistry: Singleton registry for filter classes
- register_filter: Decorator for auto-registration
"""

from __future__ import annotations
from typing import Type

from .base import BaseFilter


class FilterRegistry:
    """
    Singleton registry for filter classes.

    Maintains:
    - Hierarchical structure for UI: category -> filter_name -> filter_class
    - Flat lookup by filter_name (globally unique)
    """

    _instance: FilterRegistry | None = None

    # Preferred category order (categories not in this list appear at end, sorted)
    CATEGORY_ORDER = ["Frequency", "Spatial", "Amplitude"]

    # Preferred filter order within categories (filters not in list appear at end, sorted)
    FILTER_ORDER: dict[str, list[str]] = {
        "Amplitude": ["SEC Gain", "AGC"],  # SEC before AGC (physics-based before data-driven)
    }

    def __init__(self) -> None:
        # Structure for UI: {category: {filter_name: filter_class}}
        self._by_category: dict[str, dict[str, Type[BaseFilter]]] = {}
        # Flat lookup: {filter_name: filter_class}
        self._by_name: dict[str, Type[BaseFilter]] = {}

    @classmethod
    def get_instance(cls) -> FilterRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = FilterRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        cls._instance = None

    def register(self, filter_class: Type[BaseFilter]) -> None:
        """Register a filter class.

        Args:
            filter_class: The filter class to register
        """
        category = filter_class.category
        filter_name = filter_class.filter_name

        # Check global uniqueness of filter_name
        if filter_name in self._by_name:
            existing = self._by_name[filter_name]
            # Allow re-registration when running module with -m (new class has __main__ module)
            if existing.__name__ == filter_class.__name__ and filter_class.__module__ == "__main__":
                return
            raise ValueError(f"Filter '{filter_name}' already registered by {existing.__module__}.{existing.__name__}")

        # Add to category structure
        if category not in self._by_category:
            self._by_category[category] = {}
        self._by_category[category][filter_name] = filter_class

        # Add to flat lookup
        self._by_name[filter_name] = filter_class

    def get_categories(self) -> list[str]:
        """Get all registered categories in preferred order."""
        categories = list(self._by_category.keys())
        # Sort by CATEGORY_ORDER index, unknown categories go to end (sorted)
        def sort_key(cat: str) -> tuple[int, str]:
            if cat in self.CATEGORY_ORDER:
                return (self.CATEGORY_ORDER.index(cat), cat)
            return (len(self.CATEGORY_ORDER), cat)
        return sorted(categories, key=sort_key)

    def get_filter_names(self, category: str) -> list[str]:
        """Get filter names for a category in preferred order."""
        names = list(self._by_category.get(category, {}).keys())
        order = self.FILTER_ORDER.get(category, [])

        def sort_key(name: str) -> tuple[int, str]:
            if name in order:
                return (order.index(name), name)
            return (len(order), name)

        return sorted(names, key=sort_key)

    def get_all_filter_names(self) -> list[str]:
        """Get all registered filter names (sorted)."""
        return sorted(self._by_name.keys())

    def get_filter_class(self, filter_name: str) -> Type[BaseFilter]:
        """Get filter class by filter_name (globally unique)."""
        if filter_name not in self._by_name:
            raise KeyError(f"Unknown filter: {filter_name}")
        return self._by_name[filter_name]

    def get_filter_class_by_category(self, category: str, filter_name: str) -> Type[BaseFilter]:
        """Get filter class by category and filter_name (for UI)."""
        if category not in self._by_category:
            raise KeyError(f"Unknown category: {category}")
        if filter_name not in self._by_category[category]:
            raise KeyError(f"Unknown filter: {filter_name}")
        return self._by_category[category][filter_name]

    def create_filter(self, filter_name: str, **kwargs) -> BaseFilter:
        """Create a filter instance by filter_name."""
        filter_class = self.get_filter_class(filter_name)
        return filter_class(**kwargs)


def register_filter(cls: Type[BaseFilter]) -> Type[BaseFilter]:
    """
    Decorator to auto-register a filter class.

    Usage:
        @register_filter
        class BandpassFilter(BaseFilter):
            category = "Frequency"
            filter_name = "Bandpass"
            ...
    """
    FilterRegistry.get_instance().register(cls)
    return cls
