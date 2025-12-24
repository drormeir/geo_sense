"""
FIR (Finite Impulse Response) filter for seismic/GPR data.

Supports multiple design methods: Window, Least-squares, Parks-McClellan.
"""

from typing import Any
import numpy as np
from scipy.signal import firwin, firls, remez, lfilter

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class FIRFilter(BaseFilter):
    """
    FIR (Finite Impulse Response) filter with selectable design.

    Supported designs:
    - Window: Uses window function (Hamming, Hann, Blackman, etc.)
    - Least-squares: Minimizes least-squares error in frequency response
    - Parks-McClellan: Equiripple design, optimal in minimax sense

    FIR filters have:
    - Guaranteed stability (no feedback)
    - Linear phase (symmetric coefficients) - no phase distortion
    - More coefficients needed for sharp rolloff compared to IIR
    """

    category = "Frequency"
    filter_name = "FIR"
    description = "FIR filter with selectable design (Window, Least-squares, Parks-McClellan)"

    parameter_specs = [
        FilterParameterSpec(
            name="design",
            display_name="Design",
            param_type=ParameterType.CHOICE,
            default="window",
            choices=["window", "least-squares", "parks-mcclellan"],
            tooltip="Filter design method"
        ),
        FilterParameterSpec(
            name="window",
            display_name="Window",
            param_type=ParameterType.CHOICE,
            default="hamming",
            choices=["hamming", "hann", "blackman", "bartlett", "kaiser"],
            tooltip="Window function (Window design only)"
        ),
        FilterParameterSpec(
            name="low_freq",
            display_name="Low Cutoff",
            param_type=ParameterType.FLOAT,
            default=10.0,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="Low cutoff frequency"
        ),
        FilterParameterSpec(
            name="high_freq",
            display_name="High Cutoff",
            param_type=ParameterType.FLOAT,
            default=100.0,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="High cutoff frequency"
        ),
        FilterParameterSpec(
            name="num_taps",
            display_name="Number of Taps",
            param_type=ParameterType.INT,
            default=101,
            min_value=11,
            max_value=1001,
            step=10,
            tooltip="Filter length (odd number recommended, more taps = sharper rolloff)"
        ),
        FilterParameterSpec(
            name="filter_type",
            display_name="Type",
            param_type=ParameterType.CHOICE,
            default="bandpass",
            choices=["bandpass", "lowpass", "highpass"],
            tooltip="Filter type: bandpass, lowpass (uses high cutoff), highpass (uses low cutoff)"
        ),
    ]

    def reset_defaults_from_data(self, data_info: dict[str, Any]) -> None:
        """Reset filter parameters to sensible defaults based on data characteristics."""

        self._antenna_frequencies_hz = data_info['antenna_frequencies_hz']
        self.set_parameter("low_freq", self._antenna_frequencies_hz[0]*0.5 / 1_000_000.0)
        self.set_parameter("high_freq", self._antenna_frequencies_hz[-1]*5.0 / 1_000_000.0)


    def apply(self, data: np.ndarray|None, shape_interval: tuple[float,float]) -> tuple[np.ndarray|None, tuple[float,float]]:
        """Apply FIR filter to seismic data."""
        if data is None or data.size == 0:
            return data, shape_interval

        design = self.get_parameter("design")
        window = self.get_parameter("window")
        low_freq = self.get_parameter("low_freq")
        high_freq = self.get_parameter("high_freq")
        num_taps = self.get_parameter("num_taps")
        filter_type = self.get_parameter("filter_type")

        # Ensure odd number of taps for type I linear phase
        if num_taps % 2 == 0:
            num_taps += 1

        # Calculate Nyquist frequency
        sample_interval = shape_interval[0]
        fs = 1.0 / sample_interval
        nyquist = fs / 2.0

        # Ensure frequencies are valid
        low_freq = min(low_freq, nyquist * 0.99)
        high_freq = min(high_freq, nyquist * 0.99)

        if low_freq >= high_freq and filter_type == "bandpass":
            low_freq, high_freq = high_freq, low_freq
            if low_freq == high_freq:
                high_freq = low_freq * 1.1

        # Design filter coefficients
        if design == "window":
            coeffs = self._design_window(filter_type, low_freq, high_freq, num_taps, nyquist, window)
        elif design == "least-squares":
            coeffs = self._design_least_squares(filter_type, low_freq, high_freq, num_taps, nyquist)
        else:  # parks-mcclellan
            coeffs = self._design_remez(filter_type, low_freq, high_freq, num_taps, nyquist)

        # Apply zero-phase filtering (forward-backward)
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            # Forward filter
            filtered = lfilter(coeffs, 1.0, data[:, i])
            # Backward filter (reverse, filter, reverse)
            result[:, i] = lfilter(coeffs, 1.0, filtered[::-1])[::-1]

        return result, shape_interval


    def _design_window(self, filter_type: str, low_freq: float, high_freq: float,
                       num_taps: int, nyquist: float, window: str) -> np.ndarray:
        """Design FIR filter using window method."""
        if filter_type == "bandpass":
            return firwin(num_taps, [low_freq, high_freq], pass_zero=False,
                         fs=nyquist * 2, window=window)
        elif filter_type == "lowpass":
            return firwin(num_taps, high_freq, pass_zero=True,
                         fs=nyquist * 2, window=window)
        else:  # highpass
            return firwin(num_taps, low_freq, pass_zero=False,
                         fs=nyquist * 2, window=window)

    def _design_least_squares(self, filter_type: str, low_freq: float, high_freq: float,
                              num_taps: int, nyquist: float) -> np.ndarray:
        """Design FIR filter using least-squares method."""
        # Define frequency bands and desired response
        trans_width = min(low_freq * 0.2, (high_freq - low_freq) * 0.1, 5.0)

        if filter_type == "bandpass":
            bands = [0, low_freq - trans_width, low_freq, high_freq,
                    high_freq + trans_width, nyquist]
            desired = [0, 0, 1, 1, 0, 0]
        elif filter_type == "lowpass":
            bands = [0, high_freq, high_freq + trans_width, nyquist]
            desired = [1, 1, 0, 0]
        else:  # highpass
            bands = [0, low_freq - trans_width, low_freq, nyquist]
            desired = [0, 0, 1, 1]

        # Ensure bands are valid
        bands = [max(0, min(b, nyquist)) for b in bands]
        # Remove duplicates while preserving order
        clean_bands = [bands[0]]
        clean_desired = [desired[0]]
        for i in range(1, len(bands)):
            if bands[i] > clean_bands[-1]:
                clean_bands.append(bands[i])
                clean_desired.append(desired[i])

        return firls(num_taps, clean_bands, clean_desired, fs=nyquist * 2)

    def _design_remez(self, filter_type: str, low_freq: float, high_freq: float,
                      num_taps: int, nyquist: float) -> np.ndarray:
        """Design FIR filter using Parks-McClellan (Remez) algorithm."""
        trans_width = min(low_freq * 0.2, (high_freq - low_freq) * 0.1, 5.0)

        if filter_type == "bandpass":
            bands = [0, low_freq - trans_width, low_freq, high_freq,
                    high_freq + trans_width, nyquist]
            desired = [0, 1, 0]
        elif filter_type == "lowpass":
            bands = [0, high_freq, high_freq + trans_width, nyquist]
            desired = [1, 0]
        else:  # highpass
            bands = [0, low_freq - trans_width, low_freq, nyquist]
            desired = [0, 1]

        # Ensure bands are valid
        bands = [max(0.001, min(b, nyquist - 0.001)) for b in bands]

        try:
            return remez(num_taps, bands, desired, fs=nyquist * 2)
        except Exception:
            # Fall back to window method if remez fails
            return self._design_window(filter_type, low_freq, high_freq,
                                       num_taps, nyquist, "hamming")

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the FIR filter.

        Shows filter response comparison for different FIR designs.
        """
        from scipy.fft import fft, fftfreq

        # === Create synthetic data ===
        sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
        duration = 0.5  # 500 ms
        n_samples = int(duration / sample_interval)
        t = np.arange(n_samples) * sample_interval
        t_ms = t * 1000

        signal = (
            1.0 * np.sin(2 * np.pi * 5 * t) +
            2.0 * np.sin(2 * np.pi * 50 * t) +
            1.5 * np.sin(2 * np.pi * 80 * t) +
            0.8 * np.sin(2 * np.pi * 200 * t)
        )
        data = signal.reshape(-1, 1).astype(np.float32)

        # Filter parameters
        low_freq, high_freq = 20.0, 120.0
        num_taps = 101

        # === Get filter responses for different designs ===
        designs = ["window", "least-squares", "parks-mcclellan"]
        design_labels = ["Window (Hamming)", "Least-Squares", "Parks-McClellan"]
        colors = ["blue", "green", "red"]

        freqs = np.abs(fftfreq(n_samples, d=sample_interval))
        freq_mask = freqs <= 200
        zoom_mask = (freqs >= 5) & (freqs <= 50)

        impulse = np.zeros((n_samples, 1), dtype=np.float32)
        impulse[n_samples // 2, 0] = 1.0

        responses = {}
        filtered_results = {}
        for design in designs:
            filt = cls(low_freq=low_freq, high_freq=high_freq,
                       num_taps=num_taps, design=design)
            impulse_response, _ = filt.apply(impulse, (sample_interval,1.0))
            responses[design] = np.abs(fft(impulse_response[:, 0]))
            filtered_results[design], _ = filt.apply(data, (sample_interval,1.0))

        # Build line specs for each plot
        filtered_lines = [
            {'x': t_ms, 'y': filtered_results[d][:, 0], 'color': c,
             'linewidth': 0.8, 'alpha': 0.7, 'label': l}
            for d, l, c in zip(designs, design_labels, colors)
        ]

        response_lines = []
        for d, l, c in zip(designs, design_labels, colors):
            resp = responses[d]
            resp_norm = resp / np.max(resp) if np.max(resp) > 0 else resp
            response_lines.append(
                {'x': freqs[freq_mask], 'y': resp_norm[freq_mask],
                 'color': c, 'linewidth': 1.5, 'label': l}
            )

        zoom_lines = []
        for d, l, c in zip(designs, design_labels, colors):
            resp = responses[d]
            resp_norm = resp / np.max(resp) if np.max(resp) > 0 else resp
            zoom_lines.append(
                {'x': freqs[zoom_mask], 'y': resp_norm[zoom_mask],
                 'color': c, 'linewidth': 1.5, 'label': l}
            )

        subplots = [
            # Top-left: Original signal
            {
                'lines': [{'x': t_ms, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8}],
                'title': "Original Signal",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Top-right: Filtered signals
            {
                'lines': filtered_lines,
                'title': "Filtered Signals (All Designs)",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-left: Filter response comparison
            {
                'lines': response_lines,
                'axvlines': [
                    {'x': low_freq, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                    {'x': high_freq, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                ],
                'title': "Filter Response Comparison (All Designs)",
                'xlabel': "Frequency (Hz)",
                'ylabel': "Normalized Response",
                'legend': True,
                'grid': True,
                'ylim': (-0.05, 1.15),
            },
            # Bottom-right: Zoomed transition band
            {
                'lines': zoom_lines,
                'axvlines': [{'x': low_freq, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7}],
                'title': "Low Transition Band (Zoomed)",
                'xlabel': "Frequency (Hz)",
                'ylabel': "Normalized Response",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': f"FIR Filter Demo (Low={low_freq}Hz, High={high_freq}Hz, Taps={num_taps})",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(FIRFilter.describe())
    print()
    FIRFilter.demo()
