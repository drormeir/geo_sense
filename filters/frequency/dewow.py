"""
Dewow filter for GPR data.

Removes the low-frequency "wow" artifact caused by direct coupling
between transmitter and receiver in Ground Penetrating Radar systems.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class DewowFilter(BaseFilter):
    """
    Dewow filter for GPR data.

    The "wow" is a slowly varying, low-frequency component in GPR data
    caused by:
    - Direct coupling between transmitter and receiver antennas
    - Instrument DC drift
    - Near-field effects

    This filter removes the wow by subtracting a running mean from each
    trace, effectively acting as a high-pass filter. The window length
    controls the cutoff - longer windows remove lower frequencies.
    """

    category = "Frequency"
    filter_name = "Dewow"
    description = "Remove low-frequency wow artifact from GPR data"

    parameter_specs = [
        FilterParameterSpec(
            name="window_ns",
            display_name="Window Length",
            param_type=ParameterType.FLOAT,
            default=20.0,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            decimals=1,
            units="ns",
            tooltip="Running mean window length in nanoseconds"
        ),
        FilterParameterSpec(
            name="method",
            display_name="Method",
            param_type=ParameterType.CHOICE,
            default="mean",
            choices=["mean", "median"],
            tooltip="mean: running mean subtraction; median: running median (more robust to spikes)"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply dewow filter to GPR data."""
        window_ns = self.get_parameter("window_ns")
        method = self.get_parameter("method")

        # Convert window from ns to samples
        # sample_interval is in seconds, window_ns is in nanoseconds
        sample_interval_ns = sample_interval * 1e9
        window_samples = int(window_ns / sample_interval_ns)
        window_samples = max(3, window_samples)
        # Ensure odd window for symmetry
        if window_samples % 2 == 0:
            window_samples += 1

        nt, nx = data.shape
        result = np.zeros_like(data)
        half_window = window_samples // 2

        for trace_idx in range(nx):
            trace = data[:, trace_idx]

            if method == "median":
                # Running median (more robust but slower)
                for i in range(nt):
                    start = max(0, i - half_window)
                    end = min(nt, i + half_window + 1)
                    result[i, trace_idx] = trace[i] - np.median(trace[start:end])
            else:
                # Running mean - use cumsum for efficiency
                cumsum = np.zeros(nt + 1)
                cumsum[1:] = np.cumsum(trace)

                for i in range(nt):
                    start = max(0, i - half_window)
                    end = min(nt, i + half_window + 1)
                    window_mean = (cumsum[end] - cumsum[start]) / (end - start)
                    result[i, trace_idx] = trace[i] - window_mean

        return result


def _demo() -> None:
    """
    Visual demonstration of the Dewow filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.dewow

    Creates synthetic GPR-like data with wow artifact and shows
    before/after comparison.
    """
    import matplotlib.pyplot as plt

    # Print filter description
    print(DewowFilter.describe())
    print()

    # === Create synthetic GPR-like data ===
    sample_interval = 0.1e-9  # 0.1 ns (10 GHz sampling - typical for GPR)
    duration = 100e-9  # 100 ns total time window
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval * 1e9  # time in ns

    # Create synthetic GPR trace:
    # 1. "Wow" component - exponentially decaying low-frequency artifact
    wow = 5.0 * np.exp(-t / 30) * np.sin(2 * np.pi * 0.02 * t)

    # 2. Reflections - Ricker wavelets at different times
    def ricker(t, t0, f):
        """Ricker wavelet centered at t0 with central frequency f."""
        tau = t - t0
        # Scale factor for ns and GHz
        tau_scaled = tau * f / 1000  # Convert to appropriate units
        return (1 - 2 * np.pi**2 * tau_scaled**2) * np.exp(-np.pi**2 * tau_scaled**2)

    # Multiple reflections at different depths/times
    reflections = (
        2.0 * ricker(t, 15, 500) +   # Strong shallow reflection
        1.5 * ricker(t, 35, 400) +   # Medium reflection
        1.0 * ricker(t, 55, 350) +   # Deeper reflection
        0.7 * ricker(t, 75, 300)     # Deep, attenuated reflection
    )

    signal = wow + reflections

    # Reshape to 2D (samples x traces) - single trace
    data = signal.reshape(-1, 1).astype(np.float32)

    # === Apply dewow filter ===
    dewow = DewowFilter(window_ns=20.0)
    filtered_data = dewow.apply(data, sample_interval)

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Dewow Filter Demo (window={dewow.get_parameter('window_ns'):.0f}ns)")

    # Top-left: Original signal with wow
    axes[0, 0].plot(t, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].set_title("Original GPR Trace (with wow artifact)")
    axes[0, 0].set_xlabel("Time (ns)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: After dewow
    axes[0, 1].plot(t, filtered_data[:, 0], "g-", linewidth=0.8)
    axes[0, 1].set_title("After Dewow")
    axes[0, 1].set_xlabel("Time (ns)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Wow component only
    axes[1, 0].plot(t, wow, "r-", linewidth=1, label="Wow artifact")
    axes[1, 0].plot(t, reflections, "b-", linewidth=0.8, alpha=0.7, label="True reflections")
    axes[1, 0].set_title("Signal Components")
    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Comparison - filtered vs true reflections
    axes[1, 1].plot(t, reflections, "b-", linewidth=1, alpha=0.7, label="True reflections")
    axes[1, 1].plot(t, filtered_data[:, 0], "g--", linewidth=1, label="Dewowed signal")
    axes[1, 1].set_title("Comparison: True Reflections vs Dewowed")
    axes[1, 1].set_xlabel("Time (ns)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
