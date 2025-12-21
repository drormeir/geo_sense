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
        Ramps 0→1 from f1 to f2
        1 from f2 to f3 (passband)
        Ramps 1→0 from f3 to f4
        0 at f > f4
    """

    category = "Frequency"
    filter_name = "Ormsby"
    description = "Trapezoidal bandpass filter with four corner frequencies"

    parameter_specs = [
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

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply Ormsby bandpass filter to seismic data."""
        f1 = self.get_parameter("f1")
        f2 = self.get_parameter("f2")
        f3 = self.get_parameter("f3")
        f4 = self.get_parameter("f4")

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

        # Build trapezoidal frequency response
        response = np.zeros(nt)

        # Passband (f2 to f3): response = 1
        passband = (freqs >= f2) & (freqs <= f3)
        response[passband] = 1.0

        # Low ramp (f1 to f2): linear ramp from 0 to 1
        low_ramp = (freqs >= f1) & (freqs < f2)
        if f2 > f1:
            response[low_ramp] = (freqs[low_ramp] - f1) / (f2 - f1)

        # High ramp (f3 to f4): linear ramp from 1 to 0
        high_ramp = (freqs > f3) & (freqs <= f4)
        if f4 > f3:
            response[high_ramp] = (f4 - freqs[high_ramp]) / (f4 - f3)

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
    before/after comparison in time and frequency domains.
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

    # Composite signal with multiple frequencies:
    # - 5 Hz (below passband - will be attenuated)
    # - 30 Hz (in passband - will pass)
    # - 50 Hz (in passband - will pass)
    # - 150 Hz (above passband - will be attenuated)
    signal = (
        1.0 * np.sin(2 * np.pi * 5 * t) +    # Low frequency noise
        2.0 * np.sin(2 * np.pi * 30 * t) +   # Signal component 1
        1.5 * np.sin(2 * np.pi * 50 * t) +   # Signal component 2
        0.8 * np.sin(2 * np.pi * 150 * t)    # High frequency noise
    )

    # Reshape to 2D (samples x traces) - single trace
    data = signal.reshape(-1, 1).astype(np.float32)

    # === Apply Ormsby filter ===
    # Passband: 10-80 Hz (default parameters)
    ormsby = OrmsbyFilter()
    filtered_data = ormsby.apply(data, sample_interval)

    # === Compute frequency spectra ===
    freqs = np.abs(fftfreq(n_samples, d=sample_interval))
    original_spectrum = np.abs(fft(data[:, 0]))
    filtered_spectrum = np.abs(fft(filtered_data[:, 0]))

    # Build filter response for display
    f1, f2, f3, f4 = 5.0, 10.0, 60.0, 80.0  # default params
    response = np.zeros(n_samples)
    response[(freqs >= f2) & (freqs <= f3)] = 1.0
    low_ramp = (freqs >= f1) & (freqs < f2)
    response[low_ramp] = (freqs[low_ramp] - f1) / (f2 - f1)
    high_ramp = (freqs > f3) & (freqs <= f4)
    response[high_ramp] = (f4 - freqs[high_ramp]) / (f4 - f3)

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Ormsby Bandpass Filter Demo (F1=5, F2=10, F3=60, F4=80 Hz)")

    # Top-left: Original signal (time domain)
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Filtered signal (time domain)
    axes[0, 1].plot(t * 1000, filtered_data[:, 0], "g-", linewidth=0.8)
    axes[0, 1].set_title("Filtered Signal")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Original spectrum (frequency domain)
    max_freq = 200  # Show up to 200 Hz
    freq_mask = freqs <= max_freq
    axes[1, 0].plot(freqs[freq_mask], original_spectrum[freq_mask], "b-", linewidth=0.8)
    axes[1, 0].axvline(x=5, color="r", linestyle="--", alpha=0.5, label="5 Hz")
    axes[1, 0].axvline(x=30, color="g", linestyle="--", alpha=0.5, label="30 Hz")
    axes[1, 0].axvline(x=50, color="g", linestyle="--", alpha=0.5, label="50 Hz")
    axes[1, 0].axvline(x=150, color="r", linestyle="--", alpha=0.5, label="150 Hz")
    axes[1, 0].set_title("Original Spectrum")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Filtered spectrum + filter response
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(freqs[freq_mask], filtered_spectrum[freq_mask], "g-", linewidth=0.8, label="Filtered")
    ax2.plot(freqs[freq_mask], response[freq_mask], "r-", linewidth=1.5, alpha=0.7, label="Filter Response")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Filter Response", color="r")
    axes[1, 1].set_title("Filtered Spectrum + Filter Response")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude", color="g")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
