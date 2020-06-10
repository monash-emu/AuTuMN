from autumn.tool_kit.params import update_params


def test_update_params__no_request__expect_no_change():
    old_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {}
    expected_new_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__multiple_shallow_requests__expect_updated():
    old_params = {
        "foo": 1,
        "bank": 10,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {"foo": 2, "bank": 3}
    expected_new_params = {
        "foo": 2,
        "bank": 3,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__deep_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {"bar.baz": 3}
    expected_new_params = {
        "foo": 1,
        "bar": {"baz": 3, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {"boop[1]": 1}
    expected_new_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 1, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__end_of_array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {"boop[-1]": 3}
    expected_new_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 3],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__deep_array_request__expect_updated():
    old_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [7, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    update_request = {"bar.boop[0]": 1}
    expected_new_params = {
        "foo": 1,
        "bar": {"baz": 2, "boop": [1, 8, 9], "bing": {"bonk": 3,}},
        "boop": [4, 5, 6],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params


def test_update_params__dict_in_array_request__expect_updated():
    old_params = {
        "foo": [{"a": 1}, {"a": 2}, {"a": 3}],
    }
    update_request = {"foo[1].a": 4}
    expected_new_params = {
        "foo": [{"a": 1}, {"a": 4}, {"a": 3}],
    }
    actual_new_params = update_params(old_params, update_request)
    assert actual_new_params == expected_new_params
