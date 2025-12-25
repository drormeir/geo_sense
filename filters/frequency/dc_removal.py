"""
DC Removal filter for seismic/GPR data.

Removes the DC offset (zero-frequency component) from traces
to center the data around zero amplitude.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class DCRemovalFilter(BaseFilter):
    """
    DC Removal filter.

    Removes the mean (DC offset) from each trace to center the data
    around zero. This is a fundamental preprocessing step that:
    - Eliminates baseline drift
    - Centers waveforms for proper display
    - Prepares data for subsequent processing (e.g., FFT, correlation)
    """

    category = "Frequency"
    filter_name = "DC Removal"
    description = "Remove DC offset (mean) from traces"

    parameter_specs = [
        FilterParameterSpec(
            name="method",
            display_name="Method",
            param_type=ParameterType.CHOICE,
            default="trace",
            choices=["trace", "global", "sliding"],
            tooltip="trace: remove mean per trace; global: remove overall mean; sliding: remove running mean"
        ),
        FilterParameterSpec(
            name="window_ns",
            display_name="Sample window size (ns)",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.001,
            max_value=100.0,
            step=0.1,
            decimals=3,
            units="ns",
            tooltip="Window length for sliding method (ignored for trace/global methods)"
        ),
    ]


    def reset_defaults_from_data(self, data_info: dict[str, any]) -> None:
        """Reset filter parameters to sensible defaults based on data characteristics."""

        self._antenna_frequencies_hz = data_info['antenna_frequencies_hz']
        antenna_sample_interval_ns = 1_000_000_000.0 / self._antenna_frequencies_hz[0]
        self.set_parameter("window_ns", int(antenna_sample_interval_ns))


    def apply(self, data: np.ndarray|None, shape_interval: tuple[float,float]) -> tuple[np.ndarray|None, tuple[float,float]]:
        """Apply DC removal to seismic data."""
        if data is None or data.size == 0:
            return data, shape_interval

        sample_interval_ns = shape_interval[0] * 1e9
        method = self.get_parameter("method")

        if method == "trace":
            # Remove mean from each trace independently
            return data - np.mean(data, axis=0, keepdims=True), shape_interval

        elif method == "global":
            # Remove overall mean from entire dataset
            return data - np.mean(data), shape_interval

        else:  # sliding
            # Remove running mean (highpass-like effect)
            window_ns = self.get_parameter("window_ns")
            window_samples = int(window_ns / sample_interval_ns)
            window_samples = max(3, window_samples) | 1
            half_window = window_samples // 2

            result = np.copy(data)
            nt, nx = data.shape

            nt_range = np.arange(nt)
            start_range = np.maximum(0, nt_range - half_window)
            stop_range = np.minimum(nt, nt_range + half_window + 1)

            background = np.empty_like(data[:, 0])
            for trace_idx in range(nx):
                trace = data[:, trace_idx]
                background[:] = np.array([np.mean(trace[start:stop]) for start,stop in zip(start_range, stop_range)])
                result[:, trace_idx] -= background

            return result, shape_interval


    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the DC Removal filter.

        Creates synthetic data with DC offset and shows before/after comparison.
        """
        # === Create synthetic data ===
        sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
        duration = 0.5  # 500 ms
        n_samples = int(duration / sample_interval)
        t = np.arange(n_samples) * sample_interval
        t_ms = t * 1000

        # Create signal with:
        # - A sine wave (actual signal)
        # - A DC offset that varies (baseline drift)
        # - Some low-frequency drift
        signal_component = 1.0 * np.sin(2 * np.pi * 25 * t)
        dc_offset = 2.0  # Constant DC offset
        drift = 0.5 * np.sin(2 * np.pi * 2 * t)  # Slow drift (2 Hz)

        signal = signal_component + dc_offset + drift

        # Reshape to 2D (samples x traces) - single trace
        data = signal.reshape(-1, 1).astype(np.float32)

        # === Apply DC removal with different methods ===
        dc_trace = cls(method="trace")
        dc_sliding = cls(method="sliding", window_ms=100.0)

        filtered_trace, _ = dc_trace.apply(data, (sample_interval,1.0))
        filtered_sliding, _ = dc_sliding.apply(data, (sample_interval,1.0))

        mean_after = np.mean(filtered_trace[:, 0])

        subplots = [
            # Top-left: Original signal
            {
                'lines': [
                    {'x': t_ms, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8},
                ],
                'axhlines': [
                    {'y': 0, 'color': 'k', 'linestyle': '-', 'alpha': 0.3},
                    {'y': dc_offset, 'color': 'r', 'linestyle': '--', 'alpha': 0.7},
                ],
                'title': "Original Signal (with DC offset + drift)",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Top-right: After trace-mean removal
            {
                'lines': [
                    {'x': t_ms, 'y': filtered_trace[:, 0], 'color': 'g', 'linewidth': 0.8},
                ],
                'axhlines': [{'y': 0, 'color': 'k', 'linestyle': '-', 'alpha': 0.3}],
                'title': f"After DC Removal (method='trace', mean={mean_after:.4f})",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Bottom-left: After sliding window removal
            {
                'lines': [
                    {'x': t_ms, 'y': filtered_sliding[:, 0], 'color': 'm', 'linewidth': 0.8},
                ],
                'axhlines': [{'y': 0, 'color': 'k', 'linestyle': '-', 'alpha': 0.3}],
                'title': "After DC Removal (method='sliding', window=100ms)",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Bottom-right: Comparison of all three
            {
                'lines': [
                    {'x': t_ms, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8,
                     'alpha': 0.5, 'label': "Original"},
                    {'x': t_ms, 'y': filtered_trace[:, 0], 'color': 'g', 'linewidth': 0.8,
                     'label': "Trace mean"},
                    {'x': t_ms, 'y': filtered_sliding[:, 0], 'color': 'm', 'linewidth': 0.8,
                     'label': "Sliding (100ms)"},
                ],
                'axhlines': [{'y': 0, 'color': 'k', 'linestyle': '-', 'alpha': 0.3}],
                'title': "Comparison",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': "DC Removal Filter Demo",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(DCRemovalFilter.describe())
    print()
    DCRemovalFilter.demo()
