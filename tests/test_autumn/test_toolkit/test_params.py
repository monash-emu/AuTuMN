import os

import yaml

from autumn.utils.params import update_params, merge_dicts, load_param_file


def test_load_param_file__with_no_parent(tmpdir):
    defaults = {
        "parent": None,
        "contact_rate": 0.5,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "apples",
            "hates": "pizza",
        },
    }
    path = os.path.join(tmpdir, "default.yml")
    with open(path, "w") as f:
        yaml.dump(defaults, f)

    params = load_param_file(path)
    assert params == {
        "contact_rate": 0.5,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "apples",
            "hates": "pizza",
        },
    }


def test_load_param_file__with_parent(tmpdir):
    defaults = {
        "parent": None,
        "contact_rate": 0.5,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "apples",
            "hates": "pizza",
        },
    }
    default_path = os.path.join(tmpdir, "default.yml")
    with open(default_path, "w") as f:
        yaml.dump(defaults, f)

    scenario = {
        "parent": default_path,
        "food": {
            "favourite": "chocolate",
        },
    }
    path = os.path.join(tmpdir, "scenario.yml")
    with open(path, "w") as f:
        yaml.dump(scenario, f)

    params = load_param_file(path)
    assert params == {
        "contact_rate": 0.5,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "chocolate",
            "hates": "pizza",
        },
    }


def test_load_param_file__with_parent_and_mle(tmpdir):
    defaults = {
        "parent": None,
        "contact_rate": 0.5,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "apples",
            "hates": "pizza",
        },
    }
    default_path = os.path.join(tmpdir, "default.yml")
    with open(default_path, "w") as f:
        yaml.dump(defaults, f)

    mle = {
        "contact_rate": 0.1,
        "food.hates": "burgers",
    }
    mle_path = os.path.join(tmpdir, "mle-params.yml")
    with open(mle_path, "w") as f:
        yaml.dump(mle, f)

    scenario = {
        "parent": default_path,
        "food": {
            "favourite": "chocolate",
        },
    }
    path = os.path.join(tmpdir, "scenario.yml")
    with open(path, "w") as f:
        yaml.dump(scenario, f)

    params = load_param_file(path)
    assert params == {
        "contact_rate": 0.1,
        "ages": [0, 5, 10, 15],
        "food": {
            "favourite": "chocolate",
            "hates": "burgers",
        },
    }


def test_merge_dicts__basic_merge__with_no_key():
    base = {}
    update = {"iso3": "PHL"}
    assert merge_dicts(update, base) == {"iso3": "PHL"}


def test_merge_dicts__basic_merge__with_none_key():
    base = {"iso3": None}
    update = {"iso3": "PHL"}
    assert merge_dicts(update, base) == {"iso3": "PHL"}


def test_merge_dicts__nested_merge__with_no_key():
    base = {}
    update = {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}
    assert merge_dicts(update, base) == {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}


def test_merge_dicts__nested_merge__with_none_key():
    base = {"mixing": None}
    update = {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}
    assert merge_dicts(update, base) == {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}


def test_merge_dicts__nested_merge__with_empty_dict():
    base = {"mixing": {}}
    update = {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}
    assert merge_dicts(update, base) == {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}


def test_merge_dicts__nested_merge__with_non_conflicting_dict():
    base = {"mixing": {"baz": [7, 8, 9]}}
    update = {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}
    assert merge_dicts(update, base) == {
        "mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [7, 8, 9]}
    }


def test_merge_dicts__nested_merge__with__conflicting_dict():
    base = {"mixing": {"bar": [7, 8, 9], "baz": [7, 8, 9]}}
    update = {"mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6]}}
    assert merge_dicts(update, base) == {
        "mixing": {"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [7, 8, 9]}
    }


def test_update_params__no_request__expect_no_change():
    old_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {}
    expected_new_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__multiple_shallow_requests__expect_updated():
    old_params = {
        "foo": 1,
        "bank": 10,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {"foo": 2, "bank": 3}
    expected_new_params = {
        "foo": 2,
        "bank": 3,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__deep_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {"bar.baz": 3}
    expected_new_params = {
        "foo": 1,
        "bar": {
            "baz": 3,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {"boop(1)": 1}
    expected_new_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 1, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__end_of_array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {"boop(-1)": 3}
    expected_new_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 3],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__deep_array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [7, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    update_request = {"bar.boop(0)": 1}
    expected_new_params = {
        "foo": 1,
        "bar": {
            "baz": 2,
            "boop": [1, 8, 9],
            "bing": {
                "bonk": 3,
            },
        },
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__dict_in_array_request__expect_updated():
    old_params = {
        "foo": [{"a": 1}, {"a": 2}, {"a": 3}],
    }
    update_request = {"foo(1).a": 4}
    expected_new_params = {
        "foo": [{"a": 1}, {"a": 4}, {"a": 3}],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params
