"""
Utility functions for the UAS framework.
"""

import numpy as np

def format_value_with_units(value: float|int, units: str, decimals_digits: int=3, min_decimal_digits: int=0) -> str:
    return format_value(value, decimals_digits, min_decimal_digits) + f" [{units}]"


def format_value(value: float|int, decimals_digits: int=3, min_decimal_digits: int=0) -> str:
    if isinstance(value, int):
        ret = f"{value}"
    else:
        while True:
            ret = f"{value:.{decimals_digits}f}"
            if not ret.endswith('0'):
                break
            if decimals_digits <= min_decimal_digits:
                break
            decimals_digits -= 1
        if ret.endswith('.'):
            ret = ret[:-1]
    return ret


def simple_interpolation(vec: np.ndarray|None, x: float|None) -> float|None:
    if vec is None or vec.size == 0 or x is None:
        return None
    if x <= 0:
        return vec[0]
    if x >= vec.size - 1:
        return vec[-1]
    ix = int(x)
    dx = x - ix
    return vec[ix] * (1-dx) + vec[ix+1] * dx


def interpolate_inplace_nan_values(vec: np.ndarray) -> None:
    assert vec.ndim == 1 or vec.ndim == 2
    is_nan = np.isnan(vec)
    if vec.ndim == 2:
        is_nan = is_nan.any(axis=1)
    assert is_nan.shape == (vec.shape[0],)
    arrange_indices = np.arange(vec.shape[0])
    nan_indices = arrange_indices[is_nan]
    non_nan_indices = arrange_indices[~is_nan]
    vec_non_nan = vec[~is_nan]
    if vec.ndim == 1:
        vec[nan_indices] = np.interp(nan_indices, non_nan_indices, vec_non_nan)
    else:
        for col in range(vec.shape[1]):
            vec[nan_indices, col] = np.interp(nan_indices, non_nan_indices, vec_non_nan[:, col])
