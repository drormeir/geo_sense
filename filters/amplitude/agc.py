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

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the AGC filter.

        Creates synthetic data with varying amplitude and shows
        how AGC normalizes the signal.
        """
        from scipy.signal import hilbert

        # === Create synthetic data ===
        sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
        duration = 1.0  # 1 second
        n_samples = int(duration / sample_interval)
        t = np.arange(n_samples) * sample_interval
        t_ms = t * 1000

        # Create a signal with varying amplitude
        decay = np.exp(-2.0 * t)
        carrier = np.sin(2 * np.pi * 30 * t)

        burst = np.zeros_like(t)
        burst_center = int(0.5 * n_samples)
        burst_width = int(0.1 * n_samples)
        burst[burst_center - burst_width:burst_center + burst_width] = 3.0

        signal = (decay + burst) * carrier
        data = signal.reshape(-1, 1).astype(np.float32)

        # === Apply AGC filter ===
        agc = cls()
        filtered_data = agc.apply(data, sample_interval)
        window_ms = agc.get_parameter('window_ms')
        target_rms = agc.get_parameter('target_rms')

        # === Compute amplitude envelopes ===
        original_envelope = np.abs(hilbert(data[:, 0]))
        filtered_envelope = np.abs(hilbert(filtered_data[:, 0]))

        # === Compute running RMS ===
        window_samples = int(window_ms / (sample_interval * 1000))

        def running_rms(x, window):
            result = np.zeros_like(x)
            half_w = window // 2
            for i in range(len(x)):
                start = max(0, i - half_w)
                end = min(len(x), i + half_w + 1)
                result[i] = np.sqrt(np.mean(x[start:end] ** 2))
            return result

        original_rms = running_rms(data[:, 0], window_samples)
        filtered_rms = running_rms(filtered_data[:, 0], window_samples)

        subplots = [
            # Top-left: Original signal
            {
                'lines': [
                    {'x': t_ms, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.5},
                    {'x': t_ms, 'y': original_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ms, 'y': -original_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "Original Signal",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Top-right: After AGC
            {
                'lines': [
                    {'x': t_ms, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 0.5},
                    {'x': t_ms, 'y': filtered_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ms, 'y': -filtered_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "After AGC",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-left: Envelope comparison
            {
                'lines': [
                    {'x': t_ms, 'y': original_envelope, 'color': 'b', 'linewidth': 1, 'label': "Original"},
                    {'x': t_ms, 'y': filtered_envelope, 'color': 'g', 'linewidth': 1, 'label': "After AGC"},
                ],
                'axhlines': [{'y': target_rms, 'color': 'r', 'linestyle': '--', 'alpha': 0.7}],
                'title': "Envelope Comparison",
                'xlabel': "Time (ms)",
                'ylabel': "Envelope Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-right: Running RMS comparison
            {
                'lines': [
                    {'x': t_ms, 'y': original_rms, 'color': 'b', 'linewidth': 1, 'label': "Original RMS"},
                    {'x': t_ms, 'y': filtered_rms, 'color': 'g', 'linewidth': 1, 'label': "After AGC RMS"},
                ],
                'axhlines': [{'y': target_rms, 'color': 'r', 'linestyle': '--', 'alpha': 0.7}],
                'title': "Running RMS Comparison",
                'xlabel': "Time (ms)",
                'ylabel': "RMS Amplitude",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': f"AGC Demo (Window={window_ms:.0f}ms, Target RMS={target_rms:.1f})",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(AGCFilter.describe())
    print()
    AGCFilter.demo()
