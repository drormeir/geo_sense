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


def _demo() -> None:
    """
    Visual demonstration of the AGC filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.amplitude.agc

    Creates synthetic data with varying amplitude and shows
    how AGC normalizes the signal.
    """
    import matplotlib.pyplot as plt

    # Print filter description
    print(AGCFilter.describe())
    print()

    # === Create synthetic data ===
    sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
    duration = 1.0  # 1 second
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval

    # Create a signal with varying amplitude:
    # - Exponential decay (simulating attenuation with depth)
    # - Modulated by a sine wave (reflections)
    decay = np.exp(-2.0 * t)  # Exponential decay
    carrier = np.sin(2 * np.pi * 30 * t)  # 30 Hz carrier

    # Add a burst in the middle to show AGC handling amplitude variations
    burst = np.zeros_like(t)
    burst_center = int(0.5 * n_samples)
    burst_width = int(0.1 * n_samples)
    burst[burst_center - burst_width:burst_center + burst_width] = 3.0

    signal = (decay + burst) * carrier

    # Reshape to 2D (samples x traces) - single trace
    data = signal.reshape(-1, 1).astype(np.float32)

    # === Apply AGC filter ===
    agc = AGCFilter()
    filtered_data = agc.apply(data, sample_interval)

    # === Compute amplitude envelopes ===
    from scipy.signal import hilbert
    original_envelope = np.abs(hilbert(data[:, 0]))
    filtered_envelope = np.abs(hilbert(filtered_data[:, 0]))

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"AGC Demo (Window={agc.get_parameter('window_ms'):.0f}ms, Target RMS={agc.get_parameter('target_rms'):.1f})")

    # Top-left: Original signal (time domain)
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.5)
    axes[0, 0].plot(t * 1000, original_envelope, "r-", linewidth=1, alpha=0.7, label="Envelope")
    axes[0, 0].plot(t * 1000, -original_envelope, "r-", linewidth=1, alpha=0.7)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Filtered signal (time domain)
    axes[0, 1].plot(t * 1000, filtered_data[:, 0], "g-", linewidth=0.5)
    axes[0, 1].plot(t * 1000, filtered_envelope, "r-", linewidth=1, alpha=0.7, label="Envelope")
    axes[0, 1].plot(t * 1000, -filtered_envelope, "r-", linewidth=1, alpha=0.7)
    axes[0, 1].set_title("After AGC")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Amplitude envelope comparison
    axes[1, 0].plot(t * 1000, original_envelope, "b-", linewidth=1, label="Original")
    axes[1, 0].plot(t * 1000, filtered_envelope, "g-", linewidth=1, label="After AGC")
    axes[1, 0].axhline(y=agc.get_parameter("target_rms"), color="r", linestyle="--",
                       alpha=0.7, label=f"Target RMS={agc.get_parameter('target_rms'):.1f}")
    axes[1, 0].set_title("Envelope Comparison")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Envelope Amplitude")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Running RMS comparison
    window_samples = int(agc.get_parameter("window_ms") / (sample_interval * 1000))

    def running_rms(x, window):
        """Calculate running RMS."""
        result = np.zeros_like(x)
        half_w = window // 2
        for i in range(len(x)):
            start = max(0, i - half_w)
            end = min(len(x), i + half_w + 1)
            result[i] = np.sqrt(np.mean(x[start:end] ** 2))
        return result

    original_rms = running_rms(data[:, 0], window_samples)
    filtered_rms = running_rms(filtered_data[:, 0], window_samples)

    axes[1, 1].plot(t * 1000, original_rms, "b-", linewidth=1, label="Original RMS")
    axes[1, 1].plot(t * 1000, filtered_rms, "g-", linewidth=1, label="After AGC RMS")
    axes[1, 1].axhline(y=agc.get_parameter("target_rms"), color="r", linestyle="--",
                       alpha=0.7, label=f"Target={agc.get_parameter('target_rms'):.1f}")
    axes[1, 1].set_title("Running RMS Comparison")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("RMS Amplitude")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
