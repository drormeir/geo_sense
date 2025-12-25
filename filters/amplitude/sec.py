"""
SEC (Spherical and Exponential Compensation) gain filter.

Applies a deterministic, physics-based gain to compensate for
amplitude decay due to spherical divergence and attenuation.
"""

from typing import Any
import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class SECFilter(BaseFilter):
    """
    Spherical and Exponential Compensation (SEC) gain.

    Applies a time-varying gain to compensate for:
    - Spherical divergence: energy spreads over larger area with distance
    - Attenuation: energy absorption by the medium

    Gain is 1.0 before t0 (ground surface / second arrival), then increases:
        gain(t) = ((t - t0) / dt)^alpha * exp(beta * (t - t0))

    Where:
    - t0 = start time (typically ground surface / second arrival)
    - dt = sample interval (normalizes so gain starts at 1.0)
    - alpha = spherical divergence exponent (typically 1-2)
    - beta = exponential attenuation coefficient

    Unlike AGC, SEC preserves relative amplitude relationships
    (if parameters are correctly chosen for the medium).
    """

    category = "Amplitude"
    filter_name = "SEC Gain"
    description = "Spherical and Exponential Compensation - physics-based amplitude correction"

    parameter_specs = [
        FilterParameterSpec(
            name="alpha",
            display_name="Alpha (Spherical)",
            param_type=ParameterType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=4.0,
            step=0.1,
            decimals=2,
            tooltip="Spherical divergence exponent (0=none, 1=cylindrical, 2=spherical)"
        ),
        FilterParameterSpec(
            name="beta",
            display_name="Beta (Attenuation)",
            param_type=ParameterType.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            decimals=3,
            units="1/ns",
            tooltip="Exponential attenuation coefficient (0=none, higher=more gain at depth)"
        ),
        FilterParameterSpec(
            name="t0_ns",
            display_name="T0 (Start Time)",
            param_type=ParameterType.FLOAT,
            default=0.001,
            min_value=0.001,
            max_value=100.0,
            step=0.1,
            decimals=3,
            units="ns",
            tooltip="Time when gain starts (typically ground surface / second arrival). Gain=1.0 before this."
        ),
        FilterParameterSpec(
            name="max_gain",
            display_name="Max Gain",
            param_type=ParameterType.FLOAT,
            default=1000.0,
            min_value=1.0,
            max_value=10000.0,
            step=100.0,
            decimals=0,
            tooltip="Maximum allowed gain (clips extreme values at late times)"
        ),
    ]

    def reset_defaults_from_data(self, data_info: dict[str, Any]) -> None:
        """Reset filter parameters to sensible defaults based on data characteristics."""

        self._time_second_arrival_seconds = data_info['time_second_arrival_seconds']
        self.set_parameter("t0_ns", self._time_second_arrival_seconds*1e9)

        
    def apply(self, data: np.ndarray | None, shape_interval: tuple[float, float]) -> tuple[np.ndarray | None, tuple[float, float]]:
        """Apply SEC gain to seismic/GPR data.

        Gain starts at t0 (second arrival time) and increases with time.
        Before t0, gain is 1.0 (no amplification of direct wave).
        """
        if data is None or data.size == 0:
            return data, shape_interval

        alpha = self.get_parameter("alpha")
        beta_per_ns = self.get_parameter("beta")  # units: 1/ns
        t0_ns = self.get_parameter("t0_ns")
        max_gain = self.get_parameter("max_gain")

        # Work in nanoseconds for consistency with beta units
        sample_interval_ns = shape_interval[0] * 1e9
        nt = data.shape[0]
        t_ns = np.arange(nt) * sample_interval_ns

        # Time relative to second arrival (t0)
        t_relative_ns = t_ns - t0_ns

        # Find first sample after t0
        gain_start_idx = np.searchsorted(t_relative_ns, 0, side='right')

        # Initialize gain: 1.0 before t0
        gain = np.ones(nt)

        # Time after t0 (for gain calculation)
        t_after_t0 = t_relative_ns[gain_start_idx:]

        # Spherical divergence: ((t - t0) / dt)^alpha
        # Normalize by first sample interval so gain starts at 1.0
        dt = sample_interval_ns
        if alpha > 0:
            gain_spherical = np.power(t_after_t0 / dt, alpha)
        else:
            gain_spherical = np.ones(t_after_t0.size)

        # Exponential attenuation: exp(beta * (t - t0))
        if beta_per_ns > 0:
            gain_exponential = np.exp(beta_per_ns * t_after_t0)
        else:
            gain_exponential = np.ones(t_after_t0.size)

        # Combined gain after t0
        gain[gain_start_idx:] = gain_spherical * gain_exponential

        # Clip to max gain
        gain = np.clip(gain, 1.0, max_gain)

        # Apply gain to all traces
        result = data * gain[:, np.newaxis]

        return result, shape_interval


    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the SEC gain filter.

        Creates synthetic GPR-like data with amplitude decay and shows
        how SEC compensates for the decay.
        """
        from scipy.signal import hilbert

        # === Create synthetic GPR-like data ===
        sample_interval_ns = 0.1  # 0.1 ns (10 GHz sampling)
        duration_ns = 100  # 100 ns time window
        n_samples = int(duration_ns / sample_interval_ns)
        t_ns = np.arange(n_samples) * sample_interval_ns

        # Create synthetic reflections with natural amplitude decay
        def ricker(t, t0, f, amp):
            """Ricker wavelet centered at t0 with frequency f."""
            tau = t - t0
            tau_scaled = tau * f / 1000
            return amp * (1 - 2 * np.pi**2 * tau_scaled**2) * np.exp(-np.pi**2 * tau_scaled**2)

        # Reflections at different times with natural decay (1/t * exp(-att*t))
        decay_alpha = 1.0
        decay_beta = 0.02
        reflections = np.zeros(n_samples)

        reflection_times = [10, 25, 40, 55, 70, 85]  # ns
        reflection_amps = [1.0] * len(reflection_times)  # True amplitudes (equal)

        for t0, amp in zip(reflection_times, reflection_amps):
            # Apply natural decay to simulate real GPR data
            natural_decay = (10 / (t0 + 1)) ** decay_alpha * np.exp(-decay_beta * t0)
            reflections += ricker(t_ns, t0, 500, amp * natural_decay)

        data = reflections.reshape(-1, 1).astype(np.float32)

        # === Apply SEC filter with matching parameters ===
        sec = cls(alpha=decay_alpha, beta=decay_beta, t0_ns=1.0, max_gain=100.0)
        sample_interval_sec = sample_interval_ns * 1e-9
        filtered_data, _ = sec.apply(data, (sample_interval_sec, 1.0))

        alpha = sec.get_parameter('alpha')
        beta = sec.get_parameter('beta')
        t0_ns = sec.get_parameter('t0_ns')

        # === Compute envelopes ===
        original_envelope = np.abs(hilbert(data[:, 0]))
        filtered_envelope = np.abs(hilbert(filtered_data[:, 0]))

        # === Compute gain curve ===
        t_shifted = t_ns + t0_ns
        gain_curve = np.power(t_shifted / t0_ns, alpha) * np.exp(beta * t_ns)
        gain_curve = np.clip(gain_curve, 1.0, 100.0)

        subplots = [
            # Top-left: Original signal with decay
            {
                'lines': [
                    {'x': t_ns, 'y': data[:, 0], 'color': 'b', 'linewidth': 0.8},
                    {'x': t_ns, 'y': original_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ns, 'y': -original_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "Original Signal (with natural decay)",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Top-right: After SEC gain
            {
                'lines': [
                    {'x': t_ns, 'y': filtered_data[:, 0], 'color': 'g', 'linewidth': 0.8},
                    {'x': t_ns, 'y': filtered_envelope, 'color': 'r', 'linewidth': 1,
                     'alpha': 0.7, 'label': "Envelope"},
                    {'x': t_ns, 'y': -filtered_envelope, 'color': 'r', 'linewidth': 1, 'alpha': 0.7},
                ],
                'title': "After SEC Gain (decay compensated)",
                'xlabel': "Time (ns)",
                'ylabel': "Amplitude",
                'legend': True,
                'grid': True,
            },
            # Bottom-left: Gain curve
            {
                'lines': [
                    {'x': t_ns, 'y': gain_curve, 'color': 'purple', 'linewidth': 2},
                ],
                'title': f"SEC Gain Curve (alpha={alpha}, beta={beta})",
                'xlabel': "Time (ns)",
                'ylabel': "Gain Factor",
                'grid': True,
            },
            # Bottom-right: Envelope comparison
            {
                'lines': [
                    {'x': t_ns, 'y': original_envelope, 'color': 'b', 'linewidth': 1,
                     'label': "Original"},
                    {'x': t_ns, 'y': filtered_envelope, 'color': 'g', 'linewidth': 1,
                     'label': "After SEC"},
                ],
                'title': "Envelope Comparison",
                'xlabel': "Time (ns)",
                'ylabel': "Envelope Amplitude",
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': f"SEC Gain Demo (alpha={alpha}, beta={beta}, t0={t0_ns}ns)",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(SECFilter.describe())
    print()
    SECFilter.demo()
