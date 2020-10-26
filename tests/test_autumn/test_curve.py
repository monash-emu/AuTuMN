import pytest
import numpy as np

from autumn.curve import scale_up_function

INPUT_TIMES = list(range(100))
TEST_TIMES = np.linspace(0, 100, 66)


@pytest.mark.parametrize(
    "func_name, func",
    [
        ("linear", lambda t: 3 * t + 1),
        ("quadratic", lambda t: 2 * t ** 2 + t + 1),
        ("cubic", lambda t: t ** 3 + 2 * t ** 2 + 7),
    ],
)
@pytest.mark.parametrize("method", [1, 2, 3, 4, 5])
def test_scale_up_function(verify, method, func_name, func):
    """
    Acceptance test for scale_up_function."""
    case_str = f"scale-up-func-{func_name}-{method}"
    scaled_func = scale_up_function(x=INPUT_TIMES, y=[func(t) for t in INPUT_TIMES], method=method)
    outputs = np.zeros_like(TEST_TIMES)
    for idx in range(66):
        outputs[idx] = scaled_func(TEST_TIMES[idx])

    verify(outputs, case_str)
