"""
Entry point for running the filters package.

Usage:
    python -m filters
"""

from filters import FilterRegistry

registry = FilterRegistry.get_instance()

for category in registry.get_categories():
    print(f"{category} filters:\n")
    for name in registry.get_filter_names(category):
        filter_class = registry.get_filter_class(name)
        print(filter_class.describe())
        print()
