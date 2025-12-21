"""
Ormsby bandpass filter for seismic/GPR data.

The Ormsby filter is a trapezoidal bandpass filter defined by four
corner frequencies that create a smooth frequency response.
"""

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

    category = "Frequency"
    filter_name = "Ormsby"
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
            default=5.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="Low cut frequency - start of ramp up"
        ),
        FilterParameterSpec(
            name="f2",
            display_name="F2 (Low Pass)",
            param_type=ParameterType.FLOAT,
            default=10.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="Low pass frequency - end of ramp up"
        ),
        FilterParameterSpec(
            name="f3",
            display_name="F3 (High Pass)",
            param_type=ParameterType.FLOAT,
            default=60.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
            tooltip="High pass frequency - start of ramp down"
        ),
        FilterParameterSpec(
            name="f4",
            display_name="F4 (High Cut)",
            param_type=ParameterType.FLOAT,
            default=80.0,
            min_value=0.1,
            max_value=10000.0,
            step=1.0,
            decimals=1,
            units="Hz",
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

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply Ormsby bandpass filter to seismic data."""
        f1 = self.get_parameter("f1")
        f2 = self.get_parameter("f2")
        f3 = self.get_parameter("f3")
        f4 = self.get_parameter("f4")
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
        freqs = np.abs(fftfreq(nt, d=sample_interval))

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

        return result


def _demo() -> None:
    """
    Visual demonstration of the Ormsby filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.ormsby

    Creates synthetic data with multiple frequency components and shows
    before/after comparison with different taper functions.
    """
    import matplotlib.pyplot as plt

    # Print filter description
    print(OrmsbyFilter.describe())
    print()

    # === Create synthetic data ===
    sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
    duration = 0.5  # 500 ms
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval

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
    taper_colors = ["blue", "green", "orange", "red"]
    responses = {}

    for taper in taper_types:
        response = np.zeros(n_samples)
        response[(freqs >= f2) & (freqs <= f3)] = 1.0

        low_ramp = (freqs >= f1) & (freqs < f2)
        if f2 > f1:
            x_norm = (freqs[low_ramp] - f1) / (f2 - f1)
            response[low_ramp] = OrmsbyFilter._apply_taper(x_norm, taper)

        high_ramp = (freqs > f3) & (freqs <= f4)
        if f4 > f3:
            x_norm = (f4 - freqs[high_ramp]) / (f4 - f3)
            response[high_ramp] = OrmsbyFilter._apply_taper(x_norm, taper)

        responses[taper] = response

    # === Apply Ormsby filter with cos2 taper (default) ===
    ormsby = OrmsbyFilter()
    filtered_data = ormsby.apply(data, sample_interval)

    # === Compute spectra ===
    original_spectrum = np.abs(fft(data[:, 0]))
    filtered_spectrum = np.abs(fft(filtered_data[:, 0]))

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Ormsby Bandpass Filter Demo (F1={f1}, F2={f2}, F3={f3}, F4={f4} Hz)")

    # Top-left: Original signal (time domain)
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Filtered signal (time domain)
    axes[0, 1].plot(t * 1000, filtered_data[:, 0], "g-", linewidth=0.8)
    axes[0, 1].set_title("Filtered Signal (cos² taper)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Compare all taper responses (zoomed on transition)
    # Show low transition (f1 to f2)
    trans_mask = (freqs >= 0) & (freqs <= 100)
    for taper, color in zip(taper_types, taper_colors):
        axes[1, 0].plot(freqs[trans_mask], responses[taper][trans_mask],
                        color=color, linewidth=1.5, label=taper)
    axes[1, 0].axvline(x=f1, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].axvline(x=f2, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].axvline(x=f3, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].axvline(x=f4, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].set_title("Taper Comparison (Filter Response)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Response")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.1)

    # Bottom-right: Filtered spectrum + filter response
    max_freq = 200
    freq_mask = freqs <= max_freq
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(freqs[freq_mask], original_spectrum[freq_mask], "b-",
                    linewidth=0.8, alpha=0.5, label="Original")
    axes[1, 1].plot(freqs[freq_mask], filtered_spectrum[freq_mask], "g-",
                    linewidth=0.8, label="Filtered (cos²)")
    ax2.plot(freqs[freq_mask], responses["cos2"][freq_mask], "r-",
             linewidth=1.5, alpha=0.7, label="Filter Response")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Filter Response", color="r")
    axes[1, 1].set_title("Spectrum Before/After + Response")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
