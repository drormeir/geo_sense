"""
Entry point for running the amplitude filters package.

Usage:
    python -m filters.amplitude
"""

from filters import FilterRegistry

print("""Amplitude Filters
=================
Amplitude filters modify the amplitude/gain of seismic/GPR data without changing
the frequency content. They are used to:
- Normalize amplitude variations across traces or time
- Compensate for signal attenuation with depth/distance
- Enhance weak reflections while suppressing strong ones
- Prepare data for display with balanced amplitudes

Common applications:
- AGC (Automatic Gain Control) for uniform display amplitude
- Trace normalization for consistent trace-to-trace amplitudes
- Spherical divergence correction for geometric spreading
- Gain recovery to compensate for absorption losses
""")

registry = FilterRegistry.get_instance()
print("Available filters:\n")
for name in registry.get_filter_names("Amplitude"):
    filter_class = registry.get_filter_class(name)
    print(filter_class.describe())
    print()
