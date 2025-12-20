"""
Butterworth bandpass filter for frequency domain filtering.
"""

import numpy as np
from scipy import signal

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class BandpassFilter(BaseFilter):
    """
    Butterworth bandpass filter for frequency domain filtering.

    Applies a bandpass filter to remove frequencies outside the specified range.
    Can apply as zero-phase (forward-backward) to avoid phase distortion.
    """

    category = "Frequency"
    filter_name = "Bandpass"
    description = "Apply a Butterworth bandpass filter"

    parameter_specs = [
        FilterParameterSpec(
            name="low_freq",
            display_name="Low Frequency",
            param_type=ParameterType.FLOAT,
            default=10.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="Low cutoff frequency"
        ),
        FilterParameterSpec(
            name="high_freq",
            display_name="High Frequency",
            param_type=ParameterType.FLOAT,
            default=80.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="High cutoff frequency"
        ),
        FilterParameterSpec(
            name="order",
            display_name="Filter Order",
            param_type=ParameterType.INT,
            default=4,
            min_value=1,
            max_value=10,
            step=1,
            tooltip="Butterworth filter order (higher = steeper rolloff)"
        ),
        FilterParameterSpec(
            name="zerophase",
            display_name="Zero Phase",
            param_type=ParameterType.BOOL,
            default=True,
            tooltip="Apply filter forward and backward (no phase shift)"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply bandpass filter to seismic data."""
        low_freq = self.get_parameter("low_freq")
        high_freq = self.get_parameter("high_freq")
        order = self.get_parameter("order")
        zerophase = self.get_parameter("zerophase")

        # Calculate Nyquist frequency
        nyquist = 0.5 / sample_interval

        # Normalize frequencies
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Clamp to valid range (0 < freq < 1 for butter)
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        # Ensure low < high
        if low >= high:
            return data

        # Design filter
        b, a = signal.butter(order, [low, high], btype='band')

        # Apply to each trace
        result = np.zeros_like(data)
        nt, nx = data.shape

        for i in range(nx):
            trace = data[:, i]
            # Check for sufficient data length
            padlen = 3 * max(len(a), len(b))
            if len(trace) > padlen:
                if zerophase:
                    result[:, i] = signal.filtfilt(b, a, trace)
                else:
                    result[:, i] = signal.lfilter(b, a, trace)
            else:
                result[:, i] = trace

        return result
