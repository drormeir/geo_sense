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

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply IIR filter to seismic data."""
        low_freq = self.get_parameter("low_freq")
        high_freq = self.get_parameter("high_freq")
        order = self.get_parameter("order")
        filter_type = self.get_parameter("filter_type")
        design = self.get_parameter("design")
        ripple_db = self.get_parameter("ripple_db")
        stopband_db = self.get_parameter("stopband_db")

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

        return result


def _demo() -> None:
    """
    Visual demonstration of the IIR filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.butterworth

    Shows filter response comparison for different IIR designs.
    """
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    # Print filter description
    print(IIRFilter.describe())
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
    order = 4

    # === Get filter responses for different designs ===
    designs = ["butterworth", "bessel", "chebyshev1", "chebyshev2", "elliptic"]
    design_labels = ["Butterworth", "Bessel", "Chebyshev I", "Chebyshev II", "Elliptic"]
    colors = ["blue", "green", "orange", "red", "purple"]

    freqs = np.abs(fftfreq(n_samples, d=sample_interval))
    freq_mask = freqs <= 200

    # Get filter responses by applying to impulse
    impulse = np.zeros((n_samples, 1), dtype=np.float32)
    impulse[n_samples // 2, 0] = 1.0

    responses = {}
    filtered_results = {}
    for design in designs:
        filt = IIRFilter(low_freq=low_freq, high_freq=high_freq, order=order, design=design)
        impulse_response = filt.apply(impulse, sample_interval)
        responses[design] = np.abs(fft(impulse_response[:, 0]))
        filtered_results[design] = filt.apply(data, sample_interval)

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"IIR Filter Demo (Low={low_freq}Hz, High={high_freq}Hz, Order={order})")

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

    # Bottom-right: Zoom on transition band to show differences
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
