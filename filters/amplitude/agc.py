"""
Automatic Gain Control (AGC) filter for amplitude normalization.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class AGCFilter(BaseFilter):
    """
    Automatic Gain Control for amplitude normalization.

    Normalizes amplitude over a sliding window to enhance weak signals
    and suppress strong ones, making the display more uniform.
    """

    category = "Amplitude"
    filter_name = "AGC"
    description = "Automatic Gain Control - normalize amplitude over sliding window"

    parameter_specs = [
        FilterParameterSpec(
            name="window_ms",
            display_name="Window Length",
            param_type=ParameterType.FLOAT,
            default=500.0,
            min_value=10.0,
            max_value=5000.0,
            step=10.0,
            decimals=0,
            units="ms",
            tooltip="AGC window length in milliseconds"
        ),
        FilterParameterSpec(
            name="target_rms",
            display_name="Target RMS",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.01,
            max_value=100.0,
            step=0.1,
            decimals=2,
            tooltip="Target RMS amplitude after AGC"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply AGC to seismic data."""
        window_ms = self.get_parameter("window_ms")
        target_rms = self.get_parameter("target_rms")

        # Convert window to samples
        window_samples = int(window_ms / (sample_interval * 1000))
        window_samples = max(3, window_samples)  # Minimum 3 samples

        result = np.zeros_like(data)
        nt, nx = data.shape

        half_window = window_samples // 2

        for trace_idx in range(nx):
            trace = data[:, trace_idx]

            for i in range(nt):
                # Define window bounds
                start = max(0, i - half_window)
                end = min(nt, i + half_window + 1)

                # Calculate RMS in window
                window_data = trace[start:end]
                rms = np.sqrt(np.mean(window_data ** 2))

                # Apply gain
                if rms > 1e-10:
                    result[i, trace_idx] = trace[i] * (target_rms / rms)
                else:
                    result[i, trace_idx] = trace[i]

        return result
