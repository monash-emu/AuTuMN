import pytest
import numpy as np

from autumn.curve import scale_up_function, tanh_based_scaleup

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


def test_tanh_function():
    for grad in (0.1, 0.5, 1.0):
        for inflect_time in (0.0, 100.0, 500.0):
            for lower_asymptote in (0.0, 1.0):
                for upper_asymptote in (1.0, 100.0):

                    # Get the function
                    tanh_function = tanh_based_scaleup(
                        shape=grad,
                        inflection_time=inflect_time,
                        lower_asymptote=lower_asymptote,
                        upper_asymptote=upper_asymptote,
                    )

                    # Get the results
                    results = [tanh_function(i_time) for i_time in np.linspace(0.0, 100.0, 10)]
                    inflection_result = tanh_function(inflect_time)

                    # Make sure it makes sense
                    assert all([result >= lower_asymptote for result in results])
                    assert all([result <= upper_asymptote for result in results])
                    assert inflection_result == lower_asymptote / 2.0 + upper_asymptote / 2
