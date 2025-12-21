"""
Entry point for running the spatial filters package.

Usage:
    python -m filters.spatial
"""

from filters import FilterRegistry

print("""Spatial Filters
===============
Spatial filters operate across traces (horizontal direction in radargrams)
rather than within individual traces. They are used to:
- Remove horizontal banding and system ringing that appears on all traces
- Suppress coherent noise that is consistent across the profile
- Enhance features that vary spatially (actual subsurface reflections)
- Remove background clutter from stationary objects

Common applications:
- Background removal (BGR) to eliminate antenna ringing
- Horizontal filtering to remove flat-lying noise
- Subtracting average/median trace to enhance point reflectors
""")

registry = FilterRegistry.get_instance()
print("Available filters:\n")
for name in registry.get_filter_names("Spatial"):
    filter_class = registry.get_filter_class(name)
    print(filter_class.describe())
    print()
