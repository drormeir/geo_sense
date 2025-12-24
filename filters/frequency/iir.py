"""
IIR (Infinite Impulse Response) filter for seismic/GPR data.

Supports multiple filter designs: Butterworth, Bessel, Chebyshev, Elliptic.
"""

import numpy as np
from scipy.signal import butter, bessel, cheby1, cheby2, ellip, sosfiltfilt

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class IIRFilter(BaseFilter):
    """
    IIR (Infinite Impulse Response) filter with selectable design.

    Supported designs:
    - Butterworth: Maximally flat passband, smooth rolloff
    - Bessel: Linear phase (minimal ringing/overshoot)
    - Chebyshev I: Steeper rolloff, ripple in passband
    - Chebyshev II: Steeper rolloff, ripple in stopband
    - Elliptic: Steepest rolloff, ripple in both bands

    Zero-phase filtering (filtfilt) is used to eliminate phase distortion.
    """

    category = "Frequency"
    filter_name = "IIR"
    description = "IIR filter with selectable design (Butterworth, Bessel, Chebyshev, Elliptic)"

    parameter_specs = [
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
            tooltip="Low cutoff frequency (-3dB point)"
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
            tooltip="High cutoff frequency (-3dB point)"
        ),
        FilterParameterSpec(
            name="order",
            display_name="Order",
            param_type=ParameterType.INT,
            default=4,
            min_value=1,
            max_value=10,
            step=1,
            tooltip="Filter order (higher = steeper rolloff, more ringing)"
        ),
        FilterParameterSpec(
            name="filter_type",
            display_name="Type",
            param_type=ParameterType.CHOICE,
            default="bandpass",
            choices=["bandpass", "lowpass", "highpass"],
            tooltip="Filter type: bandpass, lowpass (uses high cutoff), highpass (uses low cutoff)"
        ),
        FilterParameterSpec(
            name="design",
            display_name="Design",
            param_type=ParameterType.CHOICE,
            default="butterworth",
            choices=["butterworth", "bessel", "chebyshev1", "chebyshev2", "elliptic"],
            tooltip="Filter design algorithm"
        ),
        FilterParameterSpec(
            name="ripple_db",
            display_name="Passband Ripple",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            decimals=1,
            units="dB",
            tooltip="Maximum ripple in passband (Chebyshev I, Elliptic only)"
        ),
        FilterParameterSpec(
            name="stopband_db",
            display_name="Stopband Attenuation",
            param_type=ParameterType.FLOAT,
            default=40.0,
            min_value=10.0,
            max_value=100.0,
            step=5.0,
            decimals=0,
            units="dB",
            tooltip="Minimum attenuation in stopband (Chebyshev II, Elliptic only)"
        ),
    ]

    def reset_defaults_from_data(self, data_info: dict[str, any]) -> None:
        """Reset filter parameters to sensible defaults based on data characteristics."""

        self._antenna_frequencies_hz = data_info['antenna_frequencies_hz']
        if self._antenna_frequencies_hz:
            self.set_parameter("low_freq", self._antenna_frequencies_hz[0]*0.5 / 1_000_000.0)
            self.set_parameter("high_freq", self._antenna_frequencies_hz[-1]*5.0 / 1_000_000.0)


    def apply(self, data: np.ndarray|None, shape_interval: tuple[float,float]) -> tuple[np.ndarray|None, tuple[float,float]]:
        """Apply IIR filter to seismic data."""
        if data is None or data.size == 0:
            return data, shape_interval

        low_freq = self.get_parameter("low_freq")
        high_freq = self.get_parameter("high_freq")
        order = self.get_parameter("order")
        filter_type = self.get_parameter("filter_type")
        design = self.get_parameter("design")
        ripple_db = self.get_parameter("ripple_db")
        stopband_db = self.get_parameter("stopband_db")

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

        # Normalize frequencies to Nyquist
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Select frequency specification based on filter type
        if filter_type == "bandpass":
            Wn = [low_norm, high_norm]
            btype = 'band'
        elif filter_type == "lowpass":
            Wn = high_norm
            btype = 'low'
        else:  # highpass
            Wn = low_norm
            btype = 'high'

        # Design filter using second-order sections (more stable)
        if design == "butterworth":
            sos = butter(order, Wn, btype=btype, output='sos')
        elif design == "bessel":
            sos = bessel(order, Wn, btype=btype, output='sos', norm='phase')
        elif design == "chebyshev1":
            sos = cheby1(order, ripple_db, Wn, btype=btype, output='sos')
        elif design == "chebyshev2":
            sos = cheby2(order, stopband_db, Wn, btype=btype, output='sos')
        else:  # elliptic
            sos = ellip(order, ripple_db, stopband_db, Wn, btype=btype, output='sos')

        # Apply zero-phase filtering to each trace
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = sosfiltfilt(sos, data[:, i])

        return result, shape_interval

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the IIR filter.

        Shows filter response comparison for different IIR designs.
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
        order = 4

        # === Get filter responses for different designs ===
        designs = ["butterworth", "bessel", "chebyshev1", "chebyshev2", "elliptic"]
        design_labels = ["Butterworth", "Bessel", "Chebyshev I", "Chebyshev II", "Elliptic"]
        colors = ["blue", "green", "orange", "red", "purple"]

        freqs = np.abs(fftfreq(n_samples, d=sample_interval))
        freq_mask = freqs <= 200
        zoom_mask = (freqs >= 5) & (freqs <= 50)

        impulse = np.zeros((n_samples, 1), dtype=np.float32)
        impulse[n_samples // 2, 0] = 1.0

        responses = {}
        filtered_results = {}
        for design in designs:
            filt = cls(low_freq=low_freq, high_freq=high_freq, order=order, design=design)
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
            'suptitle': f"IIR Filter Demo (Low={low_freq}Hz, High={high_freq}Hz, Order={order})",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(IIRFilter.describe())
    print()
    IIRFilter.demo()
