from autumn.curve import scale_up_function, tanh_based_scaleup

from apps.covid_19.model.preprocess.mixing_matrix.microdistancing import get_microdistancing_funcs
from apps.covid_19.model.parameters import MicroDistancingFunc

LOCATIONS = ["work", "home"]


def test_microdistancing__with_no_funcs():
    funcs = get_microdistancing_funcs(params={}, locations=LOCATIONS, square_mobility_effect=False)
    assert funcs["work"](0) == 1
    assert funcs["home"](0) == 1


def test_microdistancing__with_tanh_func():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "b": -0.05,
                "c": 275,
                "sigma": 0.6,
                "upper_asymptote": 1,
            },
        }
    }

    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(
        params=params, locations=LOCATIONS, square_mobility_effect=False
    )
    assert funcs["work"](0) == 1 - expect_func(0)
    assert funcs["home"](0) == 1 - expect_func(0)
    assert funcs["work"](300) == 1 - expect_func(300)
    assert funcs["home"](300) == 1 - expect_func(300)


def test_microdistancing__with_empiric_func():
    params = {
        "foo": {
            "function_type": "empiric",
            "parameters": {
                "max_effect": 0.6,
                "times": [0, 365],
                "values": [1, 100],
            },
        }
    }
    expect_func = scale_up_function([0, 365], [0.6, 60], method=4)
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(
        params=params, locations=LOCATIONS, square_mobility_effect=False
    )
    assert funcs["work"](0) == 1 - expect_func(0)
    assert funcs["home"](0) == 1 - expect_func(0)
    assert funcs["work"](300) == 1 - expect_func(300)
    assert funcs["home"](300) == 1 - expect_func(300)


def test_microdistancing__with_tanh_func_and_square_mobility_effect():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "b": -0.05,
                "c": 275,
                "sigma": 0.6,
                "upper_asymptote": 1,
            },
        }
    }

    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(
        params=params, locations=LOCATIONS, square_mobility_effect=True
    )
    assert funcs["work"](0) == (1 - expect_func(0)) ** 2
    assert funcs["home"](0) == (1 - expect_func(0)) ** 2
    assert funcs["work"](300) == (1 - expect_func(300)) ** 2
    assert funcs["home"](300) == (1 - expect_func(300)) ** 2


def test_microdistancing__with_tanh_func_and_adjuster():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "b": -0.05,
                "c": 275,
                "sigma": 0.6,
                "upper_asymptote": 1,
            },
        },
        "foo_adjuster": {
            "function_type": "empiric",
            "parameters": {
                "max_effect": 0.6,
                "times": [0, 365],
                "values": [1, 100],
            },
        },
    }
    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    expect_adj_func = scale_up_function([0, 365], [0.6, 60], method=4)
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(
        params=params, locations=LOCATIONS, square_mobility_effect=False
    )
    assert funcs["work"](0) == 1 - expect_func(0) * expect_adj_func(0)
    assert funcs["home"](0) == 1 - expect_func(0) * expect_adj_func(0)
    assert funcs["work"](300) == 1 - expect_func(300) * expect_adj_func(300)
    assert funcs["home"](300) == 1 - expect_func(300) * expect_adj_func(300)
