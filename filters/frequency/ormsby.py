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


def _test():
    """
    Test the Ormsby filter registration and creation.

    Usage:
        python -W ignore::RuntimeWarning -m filters.frequency.ormsby

    Expected output:
        Frequency filters: ['Bandpass', 'Ormsby']
        Created filter: Ormsby
        Parameters: {'f1': 5.0, 'f2': 10.0, 'f3': 60.0, 'f4': 80.0}
        Filter applied successfully, output shape: (100, 10)
    """
    from filters import FilterRegistry

    # Get registry and list available frequency filters
    registry = FilterRegistry.get_instance()
    print("Frequency filters:", registry.get_filter_names("Frequency"))

    # Create an Ormsby filter instance with default parameters
    ormsby = registry.create_filter("Ormsby")
    print("Created filter:", ormsby.filter_name)
    print("Parameters:", ormsby.parameters)

    # Test applying the filter to synthetic data
    test_data = np.random.randn(100, 10).astype(np.float32)  # 100 samples, 10 traces
    sample_interval = 0.001  # 1 ms sample interval (1000 Hz sampling rate)

    result = ormsby.apply(test_data, sample_interval)
    print("Filter applied successfully, output shape:", result.shape)


if __name__ == "__main__":
    _test()
