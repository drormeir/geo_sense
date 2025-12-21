"""
DC Removal filter for seismic/GPR data.

Removes the DC offset (zero-frequency component) from traces
to center the data around zero amplitude.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class DCRemovalFilter(BaseFilter):
    """
    DC Removal filter.

    Removes the mean (DC offset) from each trace to center the data
    around zero. This is a fundamental preprocessing step that:
    - Eliminates baseline drift
    - Centers waveforms for proper display
    - Prepares data for subsequent processing (e.g., FFT, correlation)
    """

    category = "Frequency"
    filter_name = "DC Removal"
    description = "Remove DC offset (mean) from traces"

    parameter_specs = [
        FilterParameterSpec(
            name="method",
            display_name="Method",
            param_type=ParameterType.CHOICE,
            default="trace",
            choices=["trace", "global", "sliding"],
            tooltip="trace: remove mean per trace; global: remove overall mean; sliding: remove running mean"
        ),
        FilterParameterSpec(
            name="window_ms",
            display_name="Window Length",
            param_type=ParameterType.FLOAT,
            default=100.0,
            min_value=10.0,
            max_value=5000.0,
            step=10.0,
            decimals=0,
            units="ms",
            tooltip="Window length for sliding method (ignored for trace/global methods)"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply DC removal to seismic data."""
        method = self.get_parameter("method")

        if method == "trace":
            # Remove mean from each trace independently
            return data - np.mean(data, axis=0, keepdims=True)

        elif method == "global":
            # Remove overall mean from entire dataset
            return data - np.mean(data)

        else:  # sliding
            # Remove running mean (highpass-like effect)
            window_ms = self.get_parameter("window_ms")
            window_samples = int(window_ms / (sample_interval * 1000))
            window_samples = max(3, window_samples)

            result = np.zeros_like(data)
            nt, nx = data.shape
            half_window = window_samples // 2

            for trace_idx in range(nx):
                trace = data[:, trace_idx]
                for i in range(nt):
                    start = max(0, i - half_window)
                    end = min(nt, i + half_window + 1)
                    local_mean = np.mean(trace[start:end])
                    result[i, trace_idx] = trace[i] - local_mean

            return result


def _demo() -> None:
    """
    Visual demonstration of the DC Removal filter.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.dc_removal

    Creates synthetic data with DC offset and shows before/after comparison.
    """
    import matplotlib.pyplot as plt

    # Print filter description
    print(DCRemovalFilter.describe())
    print()

    # === Create synthetic data ===
    sample_interval = 0.001  # 1 ms (1000 Hz sampling rate)
    duration = 0.5  # 500 ms
    n_samples = int(duration / sample_interval)
    t = np.arange(n_samples) * sample_interval

    # Create signal with:
    # - A sine wave (actual signal)
    # - A DC offset that varies (baseline drift)
    # - Some low-frequency drift
    signal_component = 1.0 * np.sin(2 * np.pi * 25 * t)
    dc_offset = 2.0  # Constant DC offset
    drift = 0.5 * np.sin(2 * np.pi * 2 * t)  # Slow drift (2 Hz)

    signal = signal_component + dc_offset + drift

    # Reshape to 2D (samples x traces) - single trace
    data = signal.reshape(-1, 1).astype(np.float32)

    # === Apply DC removal with different methods ===
    dc_trace = DCRemovalFilter(method="trace")
    dc_sliding = DCRemovalFilter(method="sliding", window_ms=100.0)

    filtered_trace = dc_trace.apply(data, sample_interval)
    filtered_sliding = dc_sliding.apply(data, sample_interval)

    # === Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DC Removal Filter Demo")

    # Top-left: Original signal
    axes[0, 0].plot(t * 1000, data[:, 0], "b-", linewidth=0.8)
    axes[0, 0].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[0, 0].axhline(y=dc_offset, color="r", linestyle="--", alpha=0.7, label=f"DC offset = {dc_offset}")
    axes[0, 0].set_title("Original Signal (with DC offset + drift)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: After trace-mean removal
    axes[0, 1].plot(t * 1000, filtered_trace[:, 0], "g-", linewidth=0.8)
    axes[0, 1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[0, 1].set_title("After DC Removal (method='trace')")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)
    # Show that mean is now ~0
    mean_after = np.mean(filtered_trace[:, 0])
    axes[0, 1].text(0.02, 0.98, f"Mean = {mean_after:.4f}", transform=axes[0, 1].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom-left: After sliding window removal
    axes[1, 0].plot(t * 1000, filtered_sliding[:, 0], "m-", linewidth=0.8)
    axes[1, 0].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 0].set_title("After DC Removal (method='sliding', window=100ms)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Comparison of all three
    axes[1, 1].plot(t * 1000, data[:, 0], "b-", linewidth=0.8, alpha=0.5, label="Original")
    axes[1, 1].plot(t * 1000, filtered_trace[:, 0], "g-", linewidth=0.8, label="Trace mean")
    axes[1, 1].plot(t * 1000, filtered_sliding[:, 0], "m-", linewidth=0.8, label="Sliding (100ms)")
    axes[1, 1].axhline(y=0, color="k", linestyle="-", alpha=0.3)
    axes[1, 1].set_title("Comparison")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
