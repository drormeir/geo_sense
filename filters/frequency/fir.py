"""
FIR (Finite Impulse Response) filter for seismic/GPR data.

Supports multiple design methods: Window, Least-squares, Parks-McClellan.
"""

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
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="Low cutoff frequency"
        ),
        FilterParameterSpec(
            name="high_freq",
            display_name="High Cutoff",
            param_type=ParameterType.FLOAT,
            default=100.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
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

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply FIR filter to seismic data."""
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

        return result

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


def _demo() -> None:
    """
    Visual demonstration of the FIR filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.fir

    Shows filter response comparison for different FIR designs.
    """
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    # Print filter description
    print(FIRFilter.describe())
    print()

    # === Create synthetic data ===
    sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
    duration = 0.5  # 500 ms
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval

    # Composite signal with multiple frequencies
    signal = (
        1.0 * np.sin(2 * np.pi * 5 * t) +    # Low frequency noise
        2.0 * np.sin(2 * np.pi * 50 * t) +   # Signal in passband
        1.5 * np.sin(2 * np.pi * 80 * t) +   # Signal in passband
        0.8 * np.sin(2 * np.pi * 200 * t)    # High frequency noise
    )

    # Reshape to 2D
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

    # Get filter responses by applying to impulse
    impulse = np.zeros((n_samples, 1), dtype=np.float32)
    impulse[n_samples // 2, 0] = 1.0

    responses = {}
    filtered_results = {}
    for design in designs:
        filt = FIRFilter(low_freq=low_freq, high_freq=high_freq,
                        num_taps=num_taps, design=design)
        impulse_response = filt.apply(impulse, sample_interval)
        responses[design] = np.abs(fft(impulse_response[:, 0]))
        filtered_results[design] = filt.apply(data, sample_interval)

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"FIR Filter Demo (Low={low_freq}Hz, High={high_freq}Hz, Taps={num_taps})")

    # Top-left: Original signal
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Filtered signals (all designs overlaid)
    for design, label, color in zip(designs, design_labels, colors):
        axes[0, 1].plot(t * 1000, filtered_results[design][:, 0],
                        color=color, linewidth=0.8, alpha=0.7, label=label)
    axes[0, 1].set_title("Filtered Signals (All Designs)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Compare filter responses for different designs
    for design, label, color in zip(designs, design_labels, colors):
        resp = responses[design]
        resp_norm = resp / np.max(resp) if np.max(resp) > 0 else resp
        axes[1, 0].plot(freqs[freq_mask], resp_norm[freq_mask],
                        color=color, linewidth=1.5, label=label)
    axes[1, 0].axvline(x=low_freq, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].axvline(x=high_freq, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].set_title("Filter Response Comparison (All Designs)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Normalized Response")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.15)

    # Bottom-right: Zoom on transition band
    zoom_mask = (freqs >= 5) & (freqs <= 50)
    for design, label, color in zip(designs, design_labels, colors):
        resp = responses[design]
        resp_norm = resp / np.max(resp) if np.max(resp) > 0 else resp
        axes[1, 1].plot(freqs[zoom_mask], resp_norm[zoom_mask],
                        color=color, linewidth=1.5, label=label)
    axes[1, 1].axvline(x=low_freq, color="gray", linestyle=":", alpha=0.7)
    axes[1, 1].set_title("Low Transition Band (Zoomed)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Normalized Response")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
