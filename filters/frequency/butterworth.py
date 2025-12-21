"""
Butterworth bandpass filter for seismic/GPR data.

The Butterworth filter provides a maximally flat frequency response
in the passband with a smooth rolloff controlled by filter order.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class ButterworthFilter(BaseFilter):
    """
    Butterworth bandpass filter.

    The Butterworth filter is an IIR (Infinite Impulse Response) filter
    that provides:
    - Maximally flat frequency response in the passband (no ripple)
    - Smooth monotonic rolloff in the transition band
    - Steepness controlled by filter order

    Higher orders give steeper rolloff but may introduce phase distortion
    and ringing. Zero-phase filtering (filtfilt) is used to eliminate
    phase distortion.
    """

    category = "Frequency"
    filter_name = "Butterworth"
    description = "IIR bandpass filter with maximally flat passband response"

    parameter_specs = [
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
            tooltip="Low cutoff frequency (-3dB point)"
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
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply Butterworth filter to seismic data."""
        low_freq = self.get_parameter("low_freq")
        high_freq = self.get_parameter("high_freq")
        order = self.get_parameter("order")
        filter_type = self.get_parameter("filter_type")

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

        # Normalize frequencies to Nyquist
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Design filter using second-order sections (more stable)
        if filter_type == "bandpass":
            sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
        elif filter_type == "lowpass":
            sos = butter(order, high_norm, btype='low', output='sos')
        else:  # highpass
            sos = butter(order, low_norm, btype='high', output='sos')

        # Apply zero-phase filtering to each trace
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = sosfiltfilt(sos, data[:, i])

        return result


def _demo() -> None:
    """
    Visual demonstration of the Butterworth filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.butterworth

    Shows filter response for different orders and compares to Ormsby.
    """
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    # Print filter description
    print(ButterworthFilter.describe())
    print()

    # === Create synthetic data ===
    sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
    duration = 0.5  # 500 ms
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval
    fs = 1.0 / sample_interval
    nyquist = fs / 2.0

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

    # === Apply Butterworth with different orders ===
    orders = [2, 4, 6, 8]
    filtered_results = {}
    for order in orders:
        bw = ButterworthFilter(low_freq=low_freq, high_freq=high_freq, order=order)
        filtered_results[order] = bw.apply(data, sample_interval)

    # === Compute frequency responses ===
    freqs = np.abs(fftfreq(n_samples, d=sample_interval))
    freq_mask = freqs <= 300

    # Get filter responses by applying to impulse
    impulse = np.zeros((n_samples, 1), dtype=np.float32)
    impulse[n_samples // 2, 0] = 1.0

    responses = {}
    for order in orders:
        bw = ButterworthFilter(low_freq=low_freq, high_freq=high_freq, order=order)
        impulse_response = bw.apply(impulse, sample_interval)
        responses[order] = np.abs(fft(impulse_response[:, 0]))

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Butterworth Bandpass Filter Demo (Low={low_freq}Hz, High={high_freq}Hz)")

    # Top-left: Original signal
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Filtered signal (order 4)
    axes[0, 1].plot(t * 1000, filtered_results[4][:, 0], "g-", linewidth=0.8)
    axes[0, 1].set_title("Filtered Signal (Order 4)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Compare filter responses for different orders
    colors = ["blue", "green", "orange", "red"]
    for order, color in zip(orders, colors):
        # Normalize response for display
        resp = responses[order]
        resp_norm = resp / np.max(resp) if np.max(resp) > 0 else resp
        axes[1, 0].plot(freqs[freq_mask], resp_norm[freq_mask],
                        color=color, linewidth=1.5, label=f"Order {order}")
    axes[1, 0].axvline(x=low_freq, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].axvline(x=high_freq, color="gray", linestyle=":", alpha=0.7)
    axes[1, 0].set_title("Filter Response vs Order")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Normalized Response")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.05, 1.1)

    # Bottom-right: Spectrum before/after
    original_spectrum = np.abs(fft(data[:, 0]))
    filtered_spectrum = np.abs(fft(filtered_results[4][:, 0]))

    axes[1, 1].plot(freqs[freq_mask], original_spectrum[freq_mask], "b-",
                    linewidth=0.8, alpha=0.5, label="Original")
    axes[1, 1].plot(freqs[freq_mask], filtered_spectrum[freq_mask], "g-",
                    linewidth=0.8, label="Filtered (Order 4)")
    axes[1, 1].axvline(x=low_freq, color="r", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(x=high_freq, color="r", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Spectrum Before/After")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
