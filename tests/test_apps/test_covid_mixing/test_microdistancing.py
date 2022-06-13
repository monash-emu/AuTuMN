from autumn.models.covid_19.parameters import MicroDistancingFunc
from autumn.models.sm_sir.mixing_matrix.microdistancing import get_microdistancing_funcs
from autumn.model_features.curve import scale_up_function, tanh_based_scaleup

LOCATIONS = ["work", "home"]


def test_microdistancing__with_no_funcs():
    funcs = get_microdistancing_funcs(params={}, square_mobility_effect=False, iso3="AUS")
    assert funcs["work"](0) == 1
    assert funcs["home"](0) == 1


def test_microdistancing__with_tanh_func():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "shape": 0.05,
                "inflection_time": 275,
                "start_asymptote": 1.,
                "end_asymptote": 0.6,
            },
            "locations": LOCATIONS,
        }
    }

    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(params=params, square_mobility_effect=False, iso3="AUS")
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
            "locations": LOCATIONS,
        }
    }
    expect_func = scale_up_function([0, 365], [0.6, 60], method=4)
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(params=params, square_mobility_effect=False, iso3="AUS")
    assert funcs["work"](0) == 1 - expect_func(0)
    assert funcs["home"](0) == 1 - expect_func(0)
    assert funcs["work"](300) == 1 - expect_func(300)
    assert funcs["home"](300) == 1 - expect_func(300)


def test_microdistancing__with_tanh_func_and_square_mobility_effect():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "shape": 0.05,
                "inflection_time": 275,
                "start_asymptote": 1.,
                "end_asymptote": 0.6,
            },
            "locations": LOCATIONS,
        }
    }

    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(params=params, square_mobility_effect=True, iso3="AUS")
    assert funcs["work"](0) == (1 - expect_func(0)) ** 2
    assert funcs["home"](0) == (1 - expect_func(0)) ** 2
    assert funcs["work"](300) == (1 - expect_func(300)) ** 2
    assert funcs["home"](300) == (1 - expect_func(300)) ** 2


def test_microdistancing__with_tanh_func_and_adjuster():
    params = {
        "foo": {
            "function_type": "tanh",
            "parameters": {
                "shape": 0.05,
                "inflection_time": 275,
                "start_asymptote": 1.,
                "end_asymptote": 0.6,
            },
            "locations": LOCATIONS,
        },
        "foo_adjuster": {
            "function_type": "empiric",
            "parameters": {
                "max_effect": 0.6,
                "times": [0, 365],
                "values": [1, 100],
            },
            "locations": LOCATIONS,
        },
    }
    expect_func = tanh_based_scaleup(**params["foo"]["parameters"])
    expect_adj_func = scale_up_function([0, 365], [0.6, 60], method=4)
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    funcs = get_microdistancing_funcs(params=params, square_mobility_effect=False, iso3="AUS")
    assert funcs["work"](0) == 1 - expect_func(0) * expect_adj_func(0)
    assert funcs["home"](0) == 1 - expect_func(0) * expect_adj_func(0)
    assert funcs["work"](300) == 1 - expect_func(300) * expect_adj_func(300)
    assert funcs["home"](300) == 1 - expect_func(300) * expect_adj_func(300)


def test_microdistancing__with_survey_data():
    """
    Smoke test only
    """
    params = {
        "foo": {
            "function_type": "survey",
            "parameters": {
                "effect": .5
            },
            "locations": LOCATIONS,
        },
    }
    params = {k: MicroDistancingFunc(**v) for k, v in params.items()}
    get_microdistancing_funcs(params=params, square_mobility_effect=False, iso3="AUS")
