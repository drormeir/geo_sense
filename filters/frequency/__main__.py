"""
Entry point for running the frequency filters package.

Usage:
    python -m filters.frequency
"""

from filters import FilterRegistry

print("""Frequency Filters
=================
Frequency filters operate in the frequency domain to modify the spectral content
of seismic/GPR data. They are used to:
- Remove noise at specific frequencies (e.g., power line interference at 50/60 Hz)
- Enhance signal in a frequency band of interest (bandpass filtering)
- Remove low-frequency drift (highpass) or high-frequency noise (lowpass)
- Shape the spectrum for better resolution or signal-to-noise ratio

Common applications:
- Ormsby/trapezoidal bandpass for controlled frequency rolloff
- Notch filters to remove specific interference frequencies
- Lowpass anti-alias filtering before resampling
""")

registry = FilterRegistry.get_instance()
print("Available filters:\n")
for name in registry.get_filter_names("Frequency"):
    filter_class = registry.get_filter_class(name)
    print(filter_class.describe())
    print()
