"""
Dewow filter for GPR data.

Removes the low-frequency "wow" artifact caused by direct coupling
between transmitter and receiver in Ground Penetrating Radar systems.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class DewowFilter(BaseFilter):
    """
    Dewow filter for GPR data.

    The "wow" is a slowly varying, low-frequency component in GPR data
    caused by:
    - Direct coupling between transmitter and receiver antennas
    - Instrument DC drift
    - Near-field effects

    This filter removes the wow by subtracting a running mean from each
    trace, effectively acting as a high-pass filter. The window length
    controls the cutoff - longer windows remove lower frequencies.
    """

    category = "Frequency"
    filter_name = "Dewow"
    description = "Remove low-frequency wow artifact from GPR data"

    parameter_specs = [
        FilterParameterSpec(
            name="window_ns",
            display_name="Window Length",
            param_type=ParameterType.FLOAT,
            default=20.0,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            decimals=1,
            units="ns",
            tooltip="Running mean window length in nanoseconds"
        ),
        FilterParameterSpec(
            name="method",
            display_name="Method",
            param_type=ParameterType.CHOICE,
            default="mean",
            choices=["mean", "median"],
            tooltip="mean: running mean subtraction; median: running median (more robust to spikes)"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply dewow filter to GPR data."""
        window_ns = self.get_parameter("window_ns")
        method = self.get_parameter("method")

        # Convert window from ns to samples
        # sample_interval is in seconds, window_ns is in nanoseconds
        sample_interval_ns = sample_interval * 1e9
        window_samples = int(window_ns / sample_interval_ns)
        window_samples = max(3, window_samples)
        # Ensure odd window for symmetry
        if window_samples % 2 == 0:
            window_samples += 1

        nt, nx = data.shape
        result = np.zeros_like(data)
        half_window = window_samples // 2

        for trace_idx in range(nx):
            trace = data[:, trace_idx]

            if method == "median":
                # Running median (more robust but slower)
                for i in range(nt):
                    start = max(0, i - half_window)
                    end = min(nt, i + half_window + 1)
                    result[i, trace_idx] = trace[i] - np.median(trace[start:end])
            else:
                # Running mean - use cumsum for efficiency
                cumsum = np.zeros(nt + 1)
                cumsum[1:] = np.cumsum(trace)

                for i in range(nt):
                    start = max(0, i - half_window)
                    end = min(nt, i + half_window + 1)
                    window_mean = (cumsum[end] - cumsum[start]) / (end - start)
                    result[i, trace_idx] = trace[i] - window_mean

        return result

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the Dewow filter.

        Creates synthetic GPR-like data with wow artifact and shows
        before/after comparison.
        """
        # === Create synthetic GPR-like data ===
        sample_interval = 0.1e-9  # 0.1 ns (10 GHz sampling - typical for GPR)
        duration = 100e-9  # 100 ns total time window
        n_samples = int(duration / sample_interval)
        t = np.arange(n_samples) * sample_interval * 1e9  # time in ns

        # Create synthetic GPR trace:
        # 1. "Wow" component - exponentially decaying low-frequency artifact
        wow = 5.0 * np.exp(-t / 30) * np.sin(2 * np.pi * 0.02 * t)

        # 2. Reflections - Ricker wavelets at different times
        def ricker(t, t0, f):
            """Ricker wavelet centered at t0 with central frequency f."""
            tau = t - t0
            tau_scaled = tau * f / 1000
            return (1 - 2 * np.pi**2 * tau_scaled**2) * np.exp(-np.pi**2 * tau_scaled**2)

        reflections = (
            2.0 * ricker(t, 15, 500) +
            1.5 * ricker(t, 35, 400) +
            1.0 * ricker(t, 55, 350) +
            0.7 * ricker(t, 75, 300)
        )

        signal = wow + reflections
        data = signal.reshape(-1, 1).astype(np.float32)

        # === Apply dewow filter ===
        dewow = cls(window_ns=20.0)
        filtered_data = dewow.apply(data, sample_interval)
        window_ns = dewow.get_parameter('window_ns')

        subplots = [
            # Top-left: Original signal with wow
            {
                'lines': [{'x': t, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8}],
                'title': "Original GPR Trace (with wow artifact)",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Top-right: After dewow
            {
                'lines': [{'x': t, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 0.8}],
                'title': "After Dewow",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Bottom-left: Signal components
            {
                'lines': [
                    {'x': t, 'y': wow, 'color': 'r', 'linewidth': 1, 'label': "Wow artifact"},
                    {'x': t, 'y': reflections, 'color': 'b', 'linewidth': 0.8,
                     'alpha': 0.7, 'label': "True reflections"},
                ],
                'title': "Signal Components",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-right: Comparison
            {
                'lines': [
                    {'x': t, 'y': reflections, 'color': 'b', 'linewidth': 1,
                     'alpha': 0.7, 'label': "True reflections"},
                    {'x': t, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 1,
                     'linestyle': '--', 'label': "Dewowed signal"},
                ],
                'title': "Comparison: True Reflections vs Dewowed",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': f"Dewow Filter Demo (window={window_ns:.0f}ns)",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(DewowFilter.describe())
    print()
    DewowFilter.demo()
