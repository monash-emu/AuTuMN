"""
TODO: Test more mixing matrix functionality
- test periodic interventions
- test NPI effectiveness
"""
import pandas as pd
import pytest

from autumn.models.sm_sir.mixing_matrix.macrodistancing import parse_values, update_mixing_data


@pytest.mark.parametrize(
    "vals_in, vals_out",
    (
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, ["repeat_prev"], 3], [1, 2, 2, 3]),
        ([1, 2, ["add_to_prev", 1.23], 3], [1, 2, 3.23, 3]),
        ([0.1, 0.5, ["add_to_prev_up_to_1", 0.3], 0.9], [0.1, 0.5, 0.8, 0.9]),
        ([0.1, 0.8, ["add_to_prev_up_to_1", 0.3], 0.9], [0.1, 0.8, 1, 0.9]),
        ([1, 2, ["scale_prev", 0.5], 3], [1, 2, 1, 3]),
        ([0.1, 0.95, ["scale_prev_up_to_1", 1.1], 0.5], [0.1, 0.95, 1, 0.5]),
        ([1, 2, ["scale_prev", 1.1], 3], [1, 2, 2.2, 3]),
        ([1, 2, ["scale_prev", 0.9], 3], [1, 2, 1.8, 3]),
        ([8, ["scale_prev", 0.5], ["scale_prev", 0.5], ["scale_prev", 0.5]], [8, 4, 2, 1]),
    ),
)
def test_parse_values(vals_in, vals_out):
    """
    Ensure parse values works
    """
    assert parse_values(vals_in) == vals_out


def test_update_mixing_data__with_only_mobility_data():
    """
    Ensure using no user-specified mixing params returns a mixing dict containing only mobility data.
    """

    mixing = {}
    google_mobility_values = pd.DataFrame({"work": [1.1, 1.2, 1.3, 1.4], "other_locations": [1.5, 1.6, 1.7, 1.8]})
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = update_mixing_data(
        mixing,
        google_mobility_values,
        google_mobility_days,
    )
    assert list(actual_mixing["work"].values) == [1.1, 1.2, 1.3, 1.4]
    assert list(actual_mixing["work"].index) == [0, 1, 2, 3]
    assert list(actual_mixing["other_locations"].values) == [1.5, 1.6, 1.7, 1.8]
    assert list(actual_mixing["other_locations"].index) == [0, 1, 2, 3]
    

def test_update_mixing_data__with_user_specified_values():
    """
    Ensure user-specified date overwrites/is appended to mobility data.
    """
    mixing = {
        # Expect appended with % increase accounted for.
        "work": {
            "values": [["scale_prev", 1.1], 1.6],
            "times": ([4, 5]),
            "append": True,
        },
        # Expect overwritten
        "other_locations": {
            "values": [1.55, 1.66, 1.77, 1.88, 1.99, 1.111],
            "times": ([0, 1, 2, 3, 4, 5]),
            "append": False,
        },
        # Expect added (not overwritten)
        "school": {
            "values": [1.11, 1.22, 1.33, 1.44, 1.55, 1.66],
            "times": ([0, 1, 2, 3, 4, 5]),
            "append": False,
        },
    }
    google_mobility_values = pd.DataFrame({"work": [1.1, 1.2, 1.3, 1.4], "other_locations": [1.5, 1.6, 1.7, 1.8]})
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = update_mixing_data(
        mixing,
        google_mobility_values,
        google_mobility_days,
    )
    assert list(actual_mixing["work"].values) == [1.1, 1.2, 1.3, 1.4, 1.54, 1.6]
    assert list(actual_mixing["work"].index) == [0, 1, 2, 3, 4, 5]
    assert list(actual_mixing["other_locations"].values) == [1.55, 1.66, 1.77, 1.88, 1.99, 1.111]
    assert list(actual_mixing["other_locations"].index) == [0, 1, 2, 3, 4, 5]
    assert list(actual_mixing["school"].values) == [1.11, 1.22, 1.33, 1.44, 1.55, 1.66]
    assert list(actual_mixing["school"].index) == [0, 1, 2, 3, 4, 5]
    

def test_update_mixing_data__with_user_specified_values__out_of_date():
    """
    When a user specifies mixing values where the max date is older than the latest
    mobility data, then this should still work (no crash).
    """
    mixing = {
        "school": {
            "values": [1.11, 1.22, 1.33],
            "times": ([0, 1, 2]),  # Stale date, should be up to 3
            "append": False,
        },
    }
    google_mobility_values = pd.DataFrame({"work": [1.1, 1.2, 1.3, 1.4]})
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = update_mixing_data(
        mixing,
        google_mobility_values,
        google_mobility_days,
    )

    assert list(actual_mixing["work"].values) == [1.1, 1.2, 1.3, 1.4]
    assert list(actual_mixing["work"].index) == [0, 1, 2, 3]
    assert list(actual_mixing["school"].values) == [1.11, 1.22, 1.33]
    assert list(actual_mixing["school"].index) == [0, 1, 2]


def test_update_mixing_data__with_user_specified_values__missing_data_append():
    """
    When a user specifies mixing values that should be appended to,
    and there is no Google mobility data to append to, then the app should crash.
    """

    mixing = {
        # Expect crash because of mispecified append
        "school": {
            "values": [1.11, 1.22, 1.33, 1.44],
            "times": ([0, 1, 2, 3]),
            "append": True,  # No school data to append to
        },
    }
    google_mobility_values = pd.DataFrame({"work": [1.1, 1.2, 1.3, 1.4]})
    google_mobility_days = [0, 1, 2, 3]
    with pytest.raises(AssertionError):
        update_mixing_data(
            mixing,
            google_mobility_values,
            google_mobility_days,
        )


def test_update_mixing_data__with_user_specified_values__date_clash_append():
    """
    When a user specifies mixing values that should be appended to,
    and the min appended date is less than the max Google mobility date,
    then the appended data should overwrite historical mobility data.
    """

    mixing = {
        # Expect crash because of conflicting date
        "work": {
            "values": [1.11, 1.22, 1.33],
            "times": ([3, 4, 5]),  # Conflicting lowest date, cannot append
            "append": True,
        },
    }
    npi_effectiveness_params = {}
    google_mobility_values = pd.DataFrame({"work": [1.1, 1.2, 1.3, 1.4]})
    google_mobility_days = [0, 1, 2, 3, 4]
    actual_mixing = update_mixing_data(
        mixing,
        google_mobility_values,
        google_mobility_days,
    )
    assert list(actual_mixing["work"].values) == [1.1, 1.2, 1.3, 1.11, 1.22, 1.33]
    assert list(actual_mixing["work"].index) == [0, 1, 2, 3, 4, 5]
