"""
Background Removal (BGR) filter for GPR/seismic data.

Removes horizontal banding by subtracting an average trace
computed from multiple or all traces.
"""

import numpy as np

from ..base import BaseFilter, FilterParameterSpec, ParameterType
from ..registry import register_filter


@register_filter
class BGRFilter(BaseFilter):
    """
    Background Removal (BGR) filter.

    Computes an average (or median) trace from the data and subtracts it
    from each individual trace. This removes features that are consistent
    across all traces, such as:
    - Antenna ringing (horizontal bands)
    - System noise and DC offset
    - Direct wave and ground wave

    While preserving features that vary spatially:
    - Point reflectors (hyperbolas)
    - Dipping layers
    - Lateral variations in stratigraphy
    """

    category = "Spatial"
    filter_name = "Background Removal"
    description = "Remove average trace to eliminate horizontal banding"

    parameter_specs = [
        FilterParameterSpec(
            name="method",
            display_name="Method",
            param_type=ParameterType.CHOICE,
            default="mean",
            choices=["mean", "median"],
            tooltip="mean: average trace; median: more robust to outliers"
        ),
        FilterParameterSpec(
            name="num_traces",
            display_name="Number of Traces",
            param_type=ParameterType.INT,
            default=0,
            min_value=0,
            max_value=10000,
            step=10,
            tooltip="Number of traces to use for background (0 = all traces)"
        ),
        FilterParameterSpec(
            name="trace_selection",
            display_name="Trace Selection",
            param_type=ParameterType.CHOICE,
            default="all",
            choices=["all", "first", "last", "center"],
            tooltip="Which traces to use when num_traces > 0"
        ),
    ]

    def apply(self, data: np.ndarray, sample_interval: float) -> np.ndarray:
        """Apply background removal to data."""
        method = self.get_parameter("method")
        num_traces = self.get_parameter("num_traces")
        trace_selection = self.get_parameter("trace_selection")

        nt, nx = data.shape

        # Select traces for background computation
        if num_traces <= 0 or num_traces >= nx or trace_selection == "all":
            # Use all traces
            selected_data = data
        else:
            if trace_selection == "first":
                selected_data = data[:, :num_traces]
            elif trace_selection == "last":
                selected_data = data[:, -num_traces:]
            else:  # center
                start = (nx - num_traces) // 2
                selected_data = data[:, start:start + num_traces]

        # Compute background trace
        if method == "median":
            background = np.median(selected_data, axis=1)
        else:  # mean
            background = np.mean(selected_data, axis=1)

        # Subtract background from all traces
        result = data - background[:, np.newaxis]

        return result

    @classmethod
    def demo(cls) -> None:
        """
        Visual demonstration of the Background Removal filter.

        Creates synthetic GPR-like data with horizontal banding and shows
        before/after comparison.
        """
        # === Create synthetic GPR-like data ===
        sample_interval = 0.1e-9  # 0.1 ns
        n_samples = 500  # Time samples
        n_traces = 200   # Number of traces

        t = np.arange(n_samples) * sample_interval * 1e9  # time in ns

        # Initialize data
        data = np.zeros((n_samples, n_traces), dtype=np.float32)

        # 1. Add horizontal banding (antenna ringing) - same on all traces
        ringing = (
            3.0 * np.exp(-t / 10) * np.sin(2 * np.pi * 0.1 * t) +
            1.5 * np.exp(-t / 20) * np.sin(2 * np.pi * 0.05 * t + 1)
        )
        for i in range(n_traces):
            data[:, i] += ringing

        # 2. Add a dipping reflector (should be preserved)
        for i in range(n_traces):
            reflector_time = 20 + i * 0.1  # Dipping layer
            reflector_idx = int(reflector_time / (sample_interval * 1e9))
            if 0 <= reflector_idx < n_samples - 10:
                # Add a wavelet at the reflector position
                wavelet_t = np.arange(20) * sample_interval * 1e9
                wavelet = 2.0 * np.exp(-((wavelet_t - 5) ** 2) / 2) * np.sin(2 * np.pi * 0.5 * wavelet_t)
                end_idx = min(reflector_idx + 20, n_samples)
                data[reflector_idx:end_idx, i] += wavelet[:end_idx - reflector_idx]

        # 3. Add a point reflector (hyperbola) - should be preserved
        hyperbola_x0 = n_traces // 2
        hyperbola_t0 = 35  # ns
        velocity = 0.1  # ns per trace (controls hyperbola shape)
        for i in range(n_traces):
            dx = abs(i - hyperbola_x0)
            hyperbola_time = np.sqrt(hyperbola_t0**2 + (dx * velocity * 10)**2)
            hyperbola_idx = int(hyperbola_time / (sample_interval * 1e9))
            if 0 <= hyperbola_idx < n_samples - 10:
                wavelet_t = np.arange(15) * sample_interval * 1e9
                amplitude = 1.5 * np.exp(-dx / 30)  # Amplitude decreases with offset
                wavelet = amplitude * np.exp(-((wavelet_t - 3) ** 2) / 1.5) * np.sin(2 * np.pi * 0.6 * wavelet_t)
                end_idx = min(hyperbola_idx + 15, n_samples)
                data[hyperbola_idx:end_idx, i] += wavelet[:end_idx - hyperbola_idx]

        # === Apply BGR filter ===
        bgr = cls(method="mean", num_traces=0)
        filtered_data = bgr.apply(data, sample_interval)

        # === Prepare plot specifications ===
        vmax = float(np.percentile(np.abs(data), 98))
        trace_idx = n_traces // 2  # Center trace

        subplots = [
            # Top-left: Original data
            {
                'type': 'imshow',
                'data': data,
                'extent': [0, n_traces, t[-1], t[0]],
                'cmap': 'seismic',
                'vmin': -vmax,
                'vmax': vmax,
                'colorbar': True,
                'title': "Original (with horizontal banding)",
                'xlabel': "Trace Number",
                'ylabel': "Time (ns)",
            },
            # Top-right: After BGR
            {
                'type': 'imshow',
                'data': filtered_data,
                'extent': [0, n_traces, t[-1], t[0]],
                'cmap': 'seismic',
                'vmin': -vmax,
                'vmax': vmax,
                'colorbar': True,
                'title': "After BGR (banding removed)",
                'xlabel': "Trace Number",
                'ylabel': "Time (ns)",
            },
            # Bottom-left: Background trace
            {
                'lines': [{'x': ringing, 'y': t, 'color': 'r', 'linewidth': 0.8}],
                'title': "Background Trace (removed)",
                'xlabel': "Amplitude",
                'ylabel': "Time (ns)",
                'invert_yaxis': True,
                'grid': True,
            },
            # Bottom-right: Single trace comparison
            {
                'lines': [
                    {'x': data[:, trace_idx], 'y': t, 'color': 'b', 'linewidth': 0.8,
                     'alpha': 0.7, 'label': "Original"},
                    {'x': filtered_data[:, trace_idx], 'y': t, 'color': 'g',
                     'linewidth': 0.8, 'label': "After BGR"},
                ],
                'title': f"Single Trace Comparison (trace {trace_idx})",
                'xlabel': "Amplitude",
                'ylabel': "Time (ns)",
                'invert_yaxis': True,
                'legend': True,
                'grid': True,
            },
        ]

        figure_params = {
            'suptitle': "Background Removal (BGR) Filter Demo",
            'figsize': (12, 8),
        }

        cls.render_demo_figure(subplots, figure_params)


if __name__ == "__main__":
    print(BGRFilter.describe())
    print()
    BGRFilter.demo()
