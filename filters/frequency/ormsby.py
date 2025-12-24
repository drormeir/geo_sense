"""
Ormsby bandpass filter for seismic/GPR data.

The Ormsby filter is a trapezoidal bandpass filter defined by four
corner frequencies that create a smooth frequency response.
"""

from typing import Any
import numpy as np
from scipy.fft import fft, ifft, fftfreq

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class OrmsbyFilter(BaseFilter):
    """
    Ormsby trapezoidal bandpass filter.

    Defined by four frequencies (f1, f2, f3, f4):
    - f1: Low cut frequency (start of ramp up)
    - f2: Low pass frequency (end of ramp up)
    - f3: High pass frequency (start of ramp down)
    - f4: High cut frequency (end of ramp down)

    Frequency response:
        0 at f < f1
        Ramps 0→1 from f1 to f2 (using selected taper)
        1 from f2 to f3 (passband)
        Ramps 1→0 from f3 to f4 (using selected taper)
        0 at f > f4
    """
    f1234_ratios = np.array([0.2, 0.5, 1.5, 1.8]) # ratios of f1, f2, f3, f4 to the base frequency
    category = "Frequency"
    filter_name = "Ormsby (Bandpass)"
    description = "Trapezoidal bandpass filter with four corner frequencies"

    parameter_specs = [
        FilterParameterSpec(
            name="taper",
            display_name="Taper",
            param_type=ParameterType.CHOICE,
            default="cos2",
            choices=["linear", "cos2", "hamming", "blackman"],
            tooltip="Taper function for transition bands: linear, cos² (Hann), Hamming, Blackman"
        ),
        FilterParameterSpec(
            name="f1",
            display_name="F1 (Low Cut)",
            param_type=ParameterType.FLOAT,
            default=5,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="Low cut frequency - start of ramp up"
        ),
        FilterParameterSpec(
            name="f2",
            display_name="F2 (Low Pass)",
            param_type=ParameterType.FLOAT,
            default=10,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="Low pass frequency - end of ramp up"
        ),
        FilterParameterSpec(
            name="f3",
            display_name="F3 (High Pass)",
            param_type=ParameterType.FLOAT,
            default=60,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="High pass frequency - start of ramp down"
        ),
        FilterParameterSpec(
            name="f4",
            display_name="F4 (High Cut)",
            param_type=ParameterType.FLOAT,
            default=80,
            min_value=1,
            max_value=10000,
            step=10,
            decimals=0,
            units="MHz",
            tooltip="High cut frequency - end of ramp down"
        ),
    ]

    @staticmethod
    def _apply_taper(x: np.ndarray, taper_type: str) -> np.ndarray:
        """
        Apply taper function to normalized values x in [0, 1].

        Args:
            x: Normalized position in transition band (0 at edge, 1 at passband)
            taper_type: Type of taper function

        Returns:
            Tapered values in [0, 1]
        """
        if taper_type == "linear":
            return x
        elif taper_type == "cos2":
            # Cosine squared (Hann) taper: smooth S-curve
            return 0.5 * (1 - np.cos(np.pi * x))
        elif taper_type == "hamming":
            # Hamming taper: slightly less steep than Hann
            return 0.54 - 0.46 * np.cos(np.pi * x)
        elif taper_type == "blackman":
            # Blackman taper: very smooth, more gradual rolloff
            return 0.42 - 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x)
        else:
            return x


    def reset_defaults_from_data(self, data_info: dict[str, Any]) -> None:
        """Reset filter parameters to sensible defaults based on data characteristics."""

        self._antenna_frequencies_hz = data_info['antenna_frequencies_hz']

        if self._antenna_frequencies_hz:
            base_frequency = np.array(self._antenna_frequencies_hz)
            if len(base_frequency) < 1:
                return
            if len(base_frequency) == 1:
                base_frequency = np.array([base_frequency[0]]*4)
            else:
                # use first and last frequencies as base frequencies    
                f2 = base_frequency[0] / OrmsbyFilter.f1234_ratios[1]
                f3 = base_frequency[-1] / OrmsbyFilter.f1234_ratios[2]
                base_frequency = np.array([f2, f2, f3, f3])
            frequencies_mhz = np.round(base_frequency * OrmsbyFilter.f1234_ratios / 1_000_000.0) # convert to MHz and round to int
            self.set_parameter("f1", frequencies_mhz[0])
            self.set_parameter("f2", frequencies_mhz[1])
            self.set_parameter("f3", frequencies_mhz[2])
            self.set_parameter("f4", frequencies_mhz[3])


    def apply(self, data: np.ndarray, shape_interval: tuple[float,float]) -> tuple[np.ndarray, tuple[float,float]]:
        """Apply Ormsby bandpass filter to seismic data."""
        f1 = self.get_parameter("f1")*1_000_000.0 # convert to Hz
        f2 = self.get_parameter("f2")*1_000_000.0 # convert to Hz
        f3 = self.get_parameter("f3")*1_000_000.0 # convert to Hz
        f4 = self.get_parameter("f4")*1_000_000.0 # convert to Hz
        taper = self.get_parameter("taper")

        # Ensure frequencies are in order: f1 < f2 <= f3 < f4
        while True:
            f1, f2, f3, f4 = sorted([f1, f2, f3, f4])
            if abs(f1 - f2) < 0.1:
                f2 = f1 + 0.1
                continue
            if abs(f3 - f4) < 0.1:
                f4 = f3 + 0.1
                continue
            break

        nt = data.shape[0]

        # Calculate frequency array for FFT
        freqs = np.abs(fftfreq(nt, d=shape_interval[0]))

        # Build frequency response
        response = np.zeros(nt)

        # Passband (f2 to f3): response = 1
        passband = (freqs >= f2) & (freqs <= f3)
        response[passband] = 1.0

        # Low ramp (f1 to f2): taper from 0 to 1
        low_ramp = (freqs >= f1) & (freqs < f2)
        if f2 > f1:
            x_norm = (freqs[low_ramp] - f1) / (f2 - f1)
            response[low_ramp] = self._apply_taper(x_norm, taper)

        # High ramp (f3 to f4): taper from 1 to 0
        high_ramp = (freqs > f3) & (freqs <= f4)
        if f4 > f3:
            x_norm = (f4 - freqs[high_ramp]) / (f4 - f3)
            response[high_ramp] = self._apply_taper(x_norm, taper)

        # Apply FFT to all traces at once (along axis 0 = time samples)
        spectrum = fft(data, axis=0)
        # Apply frequency response (broadcast response to all traces)
        filtered_spectrum = spectrum * response[:, np.newaxis]
        # Inverse FFT (take real part)
        result = np.real(ifft(filtered_spectrum, axis=0))

        return result, shape_interval

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the Ormsby filter.

        Creates synthetic data with multiple frequency components and shows
        before/after comparison with different taper functions.
        """
        # === Create synthetic data ===
        sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
        duration = 0.5  # 500 ms
        n_samples = int(duration / sample_interval)
        t = np.arange(n_samples) * sample_interval
        t_ms = t * 1000

        # Composite signal with multiple frequencies
        signal = (
            1.0 * np.sin(2 * np.pi * 5 * t) +    # Low frequency noise
            2.0 * np.sin(2 * np.pi * 30 * t) +   # Signal component 1
            1.5 * np.sin(2 * np.pi * 50 * t) +   # Signal component 2
            0.8 * np.sin(2 * np.pi * 150 * t)    # High frequency noise
        )

        # Reshape to 2D (samples x traces) - single trace
        data = signal.reshape(-1, 1).astype(np.float32)

        # Filter parameters
        f1, f2, f3, f4 = 5.0, 10.0, 60.0, 80.0

        # === Build filter responses for all taper types ===
        freqs = np.abs(fftfreq(n_samples, d=sample_interval))
        taper_types = ["linear", "cos2", "hamming", "blackman"]
        taper_labels = ["Linear", "Cos²", "Hamming", "Blackman"]
        taper_colors = ["blue", "green", "orange", "red"]
        responses = {}

        for taper in taper_types:
            response = np.zeros(n_samples)
            response[(freqs >= f2) & (freqs <= f3)] = 1.0

            low_ramp = (freqs >= f1) & (freqs < f2)
            if f2 > f1:
                x_norm = (freqs[low_ramp] - f1) / (f2 - f1)
                response[low_ramp] = cls._apply_taper(x_norm, taper)

            high_ramp = (freqs > f3) & (freqs <= f4)
            if f4 > f3:
                x_norm = (f4 - freqs[high_ramp]) / (f4 - f3)
                response[high_ramp] = cls._apply_taper(x_norm, taper)

            responses[taper] = response

        # === Apply Ormsby filter with cos2 taper (default) ===
        ormsby = cls()
        filtered_data = ormsby.apply(data, sample_interval)

        # === Compute spectra ===
        original_spectrum = np.abs(fft(data[:, 0]))
        filtered_spectrum = np.abs(fft(filtered_data[:, 0]))

        # Normalize spectra for comparison
        max_spectrum = max(np.max(original_spectrum), 1e-10)
        original_norm = original_spectrum / max_spectrum
        filtered_norm = filtered_spectrum / max_spectrum

        # === Build subplot specifications ===
        trans_mask = (freqs >= 0) & (freqs <= 100)
        max_freq = 200
        freq_mask = freqs <= max_freq

        # Build taper comparison lines
        taper_lines = [
            {'x': freqs[trans_mask], 'y': responses[t][trans_mask],
             'color': c, 'linewidth': 1.5, 'label': l}
            for t, l, c in zip(taper_types, taper_labels, taper_colors)
        ]

        # Build spectrum comparison lines (normalized)
        spectrum_lines = [
            {'x': freqs[freq_mask], 'y': original_norm[freq_mask], 'color': 'b',
             'linewidth': 0.8, 'alpha': 0.5, 'label': "Original"},
            {'x': freqs[freq_mask], 'y': filtered_norm[freq_mask], 'color': 'g',
             'linewidth': 0.8, 'label': "Filtered (cos²)"},
            {'x': freqs[freq_mask], 'y': responses["cos2"][freq_mask], 'color': 'r',
             'linewidth': 1.5, 'alpha': 0.7, 'linestyle': '--', 'label': "Filter Response"},
        ]

        subplots = [
            # Top-left: Original signal (time domain)
            {
                'lines': [{'x': t_ms, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8}],
                'title': "Original Signal",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Top-right: Filtered signal (time domain)
            {
                'lines': [{'x': t_ms, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 0.8}],
                'title': "Filtered Signal (cos² taper)",
                'xlabel': "Time (ms)",
                'ylabel': "Amplitude",
                'grid': True,
            },
            # Bottom-left: Compare all taper responses
            {
                'lines': taper_lines,
                'axvlines': [
                    {'x': f1, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                    {'x': f2, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                    {'x': f3, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                    {'x': f4, 'color': 'gray', 'linestyle': ':', 'alpha': 0.7},
                ],
                'title': "Taper Comparison (Filter Response)",
                'xlabel': "Frequency (Hz)",
                'ylabel': "Response",
                'legend': True,
                'grid': True,
                'ylim': (-0.05, 1.1),
            },
            # Bottom-right: Spectrum before/after + filter response (normalized)
            {
                'lines': spectrum_lines,
                'title': "Spectrum Before/After + Response (Normalized)",
                'xlabel': "Frequency (Hz)",
                'ylabel': "Normalized Magnitude / Response",
                'legend': True,
                'grid': True,
                'ylim': (-0.05, 1.15),
            },
        ]

        figure_params = {
            'suptitle': f"Ormsby Bandpass Filter Demo (F1={f1}, F2={f2}, F3={f3}, F4={f4} Hz)",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(OrmsbyFilter.describe())
    print()
    OrmsbyFilter.demo()
