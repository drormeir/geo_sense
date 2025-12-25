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
            name="window_ns",
            display_name="Window Length",
            param_type=ParameterType.FLOAT,
            default=0.1,
            min_value=0.001,
            max_value=100.0,
            step=0.1,
            decimals=3,
            units="ns",
            tooltip="AGC window length in nanoseconds"
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

    def apply(self, data: np.ndarray|None, shape_interval: tuple[float,float]) -> tuple[np.ndarray|None, tuple[float,float]]:
        """Apply AGC to seismic data."""
        if data is None or data.size == 0:
            return data, shape_interval

        window_ns = self.get_parameter("window_ns")
        target_rms = self.get_parameter("target_rms")

        # Convert window from ns to samples
        sample_interval_ns = shape_interval[0] * 1e9
        window_samples = int(window_ns / sample_interval_ns)
        window_samples = max(3, window_samples) | 1 # Minimum 3 samples
        half_window = window_samples // 2

        result = np.copy(data)
        nt, nx = data.shape


        nt_range = np.arange(nt)
        start_range = np.maximum(0, nt_range - half_window)
        stop_range = np.minimum(nt, nt_range + half_window + 1)

        rms = np.empty_like(data[:, 0])
        for trace_idx in range(nx):
            trace = data[:, trace_idx]
            cumsum = np.concatenate(([0], np.cumsum(trace ** 2)))
            rms[:] = np.array([np.sqrt((cumsum[stop] - cumsum[start]) / (stop - start)) for start,stop in zip(start_range, stop_range)])
            rms[rms < 1e-10] = target_rms
            result[:, trace_idx] *= (target_rms / rms)
        return result, shape_interval


    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the AGC filter.

        Creates synthetic data with varying amplitude and shows
        how AGC normalizes the signal.
        """
        from scipy.signal import hilbert

        # === Create synthetic data ===
        sample_interval_ns = 1  # 1 ns (1 GHz sampling rate)
        duration_ns = 1000*sample_interval_ns  # 1 microsecond
        n_samples = int(duration_ns / sample_interval_ns)
        t_ns = np.arange(n_samples) * sample_interval_ns

        # Create a signal with varying amplitude
        decay = np.exp(-2.0 * t_ns)
        carrier = np.sin(2 * np.pi * 30 * t_ns)

        burst = np.zeros_like(t_ns)
        burst_center = int(0.5 * n_samples)
        burst_width = int(0.1 * n_samples)
        burst[burst_center - burst_width:burst_center + burst_width] = 3.0

        signal = (decay + burst) * carrier
        data = signal.reshape(-1, 1).astype(np.float32)

        # === Apply AGC filter ===
        agc = cls(window_ns=0.1, target_rms=1.0)
        sample_interval_sec = sample_interval_ns * 1e-9
        filtered_data, _ = agc.apply(data, (sample_interval_sec, 1))
        window_ns = agc.get_parameter('window_ns')
        target_rms = agc.get_parameter('target_rms')

        # === Compute amplitude envelopes ===
        original_envelope = np.abs(hilbert(data[:, 0]))
        filtered_envelope = np.abs(hilbert(filtered_data[:, 0]))

        # === Compute running RMS ===
        window_samples = int(window_ns / sample_interval_ns)
        window_samples = max(3, window_samples) | 1 # Minimum 3 samples

        def running_rms(trace, start_range, stop_range):
            cumsum = np.concatenate(([0], np.cumsum(trace ** 2)))
            return np.array([np.sqrt((cumsum[stop] - cumsum[start]) / (stop - start)) for start,stop in zip(start_range, stop_range)])

        half_w = window_samples // 2
        array_range = np.arange(data[:, 0].size)
        start_range = np.maximum(0, array_range - half_w)
        stop_range = np.minimum(array_range.size, array_range + half_w + 1)


        original_rms = running_rms(data[:, 0], start_range, stop_range)
        filtered_rms = running_rms(filtered_data[:, 0], start_range, stop_range)

        subplots = [
            # Top-left: Original signal
            {
                'lines': [
                    {'x': t_ns, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.5},
                    {'x': t_ns, 'y': original_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ns, 'y': -original_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "Original Signal",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Top-right: After AGC
            {
                'lines': [
                    {'x': t_ns, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 0.5},
                    {'x': t_ns, 'y': filtered_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ns, 'y': -filtered_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "After AGC",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-left: Envelope comparison
            {
                'lines': [
                    {'x': t_ns, 'y': original_envelope, 'color': 'b', 'linewidth': 1, 'label': "Original"},
                    {'x': t_ns, 'y': filtered_envelope, 'color': 'g', 'linewidth': 1, 'label': "After AGC"},
                ],
                'axhlines': [{'y': target_rms, 'color': 'r', 'linestyle': '--', 'alpha': 0.7}],
                'title': "Envelope Comparison",
                'xlabel': "Time (ns)",
                'ylabel': "Envelope Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-right: Running RMS comparison
            {
                'lines': [
                    {'x': t_ns, 'y': original_rms, 'color': 'b', 'linewidth': 1, 'label': "Original RMS"},
                    {'x': t_ns, 'y': filtered_rms, 'color': 'g', 'linewidth': 1, 'label': "After AGC RMS"},
                ],
                'axhlines': [{'y': target_rms, 'color': 'r', 'linestyle': '--', 'alpha': 0.7}],
                'title': "Running RMS Comparison",
                'xlabel': "Time (ns)",
                'ylabel': "RMS Amplitude",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': f"AGC Demo (Window={window_ns:.0f}ns, Target RMS={target_rms:.1f})",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(AGCFilter.describe())
    print()
    AGCFilter.demo()
