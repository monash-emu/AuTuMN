"""
TODO: Test more mixing matrix functionality
- test periodic interventions
- test age adjustments
- test NPI effectiveness
- test microdistancing
"""
from datetime import date, timedelta, datetime
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from apps.covid_19.model.preprocess import mixing_matrix
from apps.covid_19.model.preprocess.mixing_matrix import adjust_location
from apps.covid_19.model.preprocess.mixing_matrix.utils import BASE_DATE
from apps.covid_19.model.parameters import Mobility, Country, MixingLocation

from autumn.inputs.social_mixing.queries import (
    get_mixing_matrix_specific_agegroups,
    get_country_mixing_matrix,
)


@pytest.mark.parametrize(
    "vals_in, vals_out",
    (
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, ["repeat_prev"], 3], [1, 2, 2, 3]),
        ([1, 2, ["add_to_prev", 1.23], 3], [1, 2, 3.23, 3]),
        ([0.1, 0.5, ["add_to_prev_up_to_1", 0.3], 0.9], [0.1, 0.5, 0.8, 0.9]),
        ([0.1, 0.8, ["add_to_prev_up_to_1", 0.3], 0.9], [0.1, 0.8, 1, 0.9]),
        ([1, 2, ["scale_prev", -1], 3], [1, 2, 0, 3]),
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
    assert adjust_location.parse_values(vals_in) == vals_out


def test_update_mixing_data__with_only_mobility_data():
    """
    Ensure using no user-specified mixing params returns a mixing dict containing only mobility data.
    """
    mixing = {}
    npi_effectiveness_params = {}
    google_mobility_values = {"work": [1.1, 1.2, 1.3, 1.4], "other_locations": [1.5, 1.6, 1.7, 1.8]}
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = adjust_location.update_mixing_data(
        mixing, npi_effectiveness_params, google_mobility_values, google_mobility_days,
    )
    assert actual_mixing == {
        "work": {"values": [1.1, 1.2, 1.3, 1.4], "times": [0, 1, 2, 3],},
        "other_locations": {"values": [1.5, 1.6, 1.7, 1.8], "times": [0, 1, 2, 3],},
    }


def test_update_mixing_data__with_user_specified_values():
    """
    Ensure user-specified date overwrites/is appended to mobility data.
    """
    mixing = {
        # Expect appended with % increase accounted for.
        "work": {"values": [["scale_prev", 1.1], 1.6], "times": ([4, 5]), "append": True,},
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
    npi_effectiveness_params = {}
    google_mobility_values = {"work": [1.1, 1.2, 1.3, 1.4], "other_locations": [1.5, 1.6, 1.7, 1.8]}
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = adjust_location.update_mixing_data(
        mixing, npi_effectiveness_params, google_mobility_values, google_mobility_days,
    )
    assert actual_mixing == {
        "work": {"values": [1.1, 1.2, 1.3, 1.4, 1.54, 1.6], "times": [0, 1, 2, 3, 4, 5]},
        "other_locations": {
            "values": [1.55, 1.66, 1.77, 1.88, 1.99, 1.111],
            "times": [0, 1, 2, 3, 4, 5],
        },
        "school": {"values": [1.11, 1.22, 1.33, 1.44, 1.55, 1.66], "times": [0, 1, 2, 3, 4, 5]},
    }


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
    npi_effectiveness_params = {}
    google_mobility_values = {"work": [1.1, 1.2, 1.3, 1.4]}
    google_mobility_days = [0, 1, 2, 3]
    actual_mixing = adjust_location.update_mixing_data(
        mixing, npi_effectiveness_params, google_mobility_values, google_mobility_days,
    )
    assert actual_mixing == {
        "work": {"values": [1.1, 1.2, 1.3, 1.4], "times": [0, 1, 2, 3]},
        "school": {"values": [1.11, 1.22, 1.33], "times": [0, 1, 2]},
    }


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
    npi_effectiveness_params = {}
    google_mobility_values = {"work": [1.1, 1.2, 1.3, 1.4]}
    google_mobility_days = [0, 1, 2, 3]
    is_periodic_intervention = False
    periodic_int_params = None
    periodic_end_time = None
    with pytest.raises(ValueError):
        adjust_location.update_mixing_data(
            mixing, npi_effectiveness_params, google_mobility_values, google_mobility_days,
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
    google_mobility_values = {"work": [1.1, 1.2, 1.3, 1.4]}
    google_mobility_days = [0, 1, 2, 3, 4]
    actual_mixing = adjust_location.update_mixing_data(
        mixing, npi_effectiveness_params, google_mobility_values, google_mobility_days,
    )
    assert actual_mixing == {
        "work": {"values": [1.1, 1.2, 1.3, 1.11, 1.22, 1.33], "times": [0, 1, 2, 3, 4, 5]},
    }


def test_build_static__for_australia():
    """
    Ensure the correct static mixing matrix for australia is produced
    """
    mm = mixing_matrix.build_static("AUS")
    assert_arr_is_close(mm, AUS_MIXING_MATRIX)


def test_build_dynamic__with_no_changes():
    """
    Ensure dynamic mixing matrix has no change over time, if no changes are supplied.
    """
    google_mobility_locations = {}
    mixing_params = {}
    npi_effectiveness_params = {}
    microdistancing_params = {}
    smooth_google_data = False
    microdistancing_locations = ["home", "other_locations", "school", "work"]
    mm_func = mixing_matrix.build_dynamic(
        country=Country(iso3="AUS"),
        mobility=Mobility(
            region=None,
            mixing=mixing_params,
            npi_effectiveness={},
            google_mobility_locations=google_mobility_locations,
            smooth_google_data=smooth_google_data,
            microdistancing=None,
            microdistancing_locations=microdistancing_locations,
        ),
    )
    mm = mm_func(0)
    assert_arr_is_close(mm, AUS_MIXING_MATRIX)

    mm = mm_func(111)
    assert_arr_is_close(mm, AUS_MIXING_MATRIX)


def test_build_dynamic__with_mobility_data(monkeypatch):
    """
    Ensure dynamic mixing matrix has no change over time, if no changes are supplied.
    """

    def get_test_mobility_data(country_iso3, region, base_date, google_mobility_locations):
        assert country_iso3 == "AUS"
        assert not region
        assert base_date
        assert google_mobility_locations == {"work": ["workplace"]}
        mobility_days = [1, 2, 3, 4, 5, 6, 7, 8]
        loc_mobility_values = {"work": [1, 1, 1, 1.5, 1.5, 1.5, 0.5, 0.5]}
        return loc_mobility_values, mobility_days

    monkeypatch.setattr(adjust_location, "get_mobility_data", get_test_mobility_data)

    microdistancing_locations = ["home", "other_locations", "school", "work"]
    google_mobility_locations = {"work": ["workplace"]}
    mixing_params = {}
    npi_effectiveness_params = {}
    smooth_google_data = False
    mm_func = mixing_matrix.build_dynamic(
        country=Country(iso3="AUS"),
        mobility=Mobility(
            region=None,
            mixing=mixing_params,
            npi_effectiveness={},
            google_mobility_locations=google_mobility_locations,
            smooth_google_data=False,
            microdistancing=None,
            microdistancing_locations=microdistancing_locations,
        ),
    )

    # Work mixing adjustment should be 1, expect no change
    mm = mm_func(0)
    assert_arr_is_close(mm, AUS_MIXING_MATRIX)

    # Work mixing adjustment should be 1.5, expect +50% work change
    mm = mm_func(5)
    expected_mm = AUS_MIXING_MATRIX + 0.5 * AUS_WORK_MIXING_MATRIX
    assert_arr_is_close(mm, expected_mm)

    # Work mixing adjustment should be 0.5, expect -50% work change
    mm = mm_func(9)
    expected_mm = AUS_MIXING_MATRIX - 0.5 * AUS_WORK_MIXING_MATRIX
    assert_arr_is_close(mm, expected_mm)


def test_build_dynamic__smoke_test():
    """
    Smoke test with typical input data.
    Doesn't actually verify anything.
    """
    microdistancing_locations = ["home", "other_locations", "school", "work"]
    google_mobility_locations = {
        "work": ["workplaces"],
        "other_locations": [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "parks",
            "transit_stations",
        ],
    }
    mixing_params = {
        "other_locations": MixingLocation(
            append=True,
            times=get_date_from_now([10, 20, 30, 40, 50]),
            values=[1, 0.4, 0.3, 0.3, 0.5],
        ),
        "work": MixingLocation(
            append=True,
            times=get_date_from_now([10, 20, 30, 40, 50]),
            values=[1, 0.9, 0.5, 0.3, 0.6],
        ),
        "school": MixingLocation(append=False, times=get_date_from_now([10, 20]), values=[1, 0]),
    }
    mm_func = mixing_matrix.build_dynamic(
        country=Country(iso3="MYS"),
        mobility=Mobility(
            region=None,
            mixing=mixing_params,
            npi_effectiveness={},
            google_mobility_locations=google_mobility_locations,
            smooth_google_data=True,
            microdistancing=None,
            microdistancing_locations=microdistancing_locations,
        ),
    )
    mm = mm_func(50)
    assert mm.shape == (16, 16)


def test_age_mixing_matrix_variable_agegroups__smoke_test():
    requested_age_breaks = [0, 20, 50]
    out_matrix = get_mixing_matrix_specific_agegroups("AUS", requested_age_breaks)
    assert out_matrix.shape == (3, 3)


def test_age_mixing_matrix_variable_agegroups__conservation():
    requested_age_breaks = [i * 5.0 for i in range(16)]  # same as original Prem age groups
    prem_matrix = get_country_mixing_matrix("all_locations", "AUS")
    out_matrix = get_mixing_matrix_specific_agegroups("AUS", requested_age_breaks)
    assert np.array_equal(out_matrix, prem_matrix)


def assert_arr_is_close(arr_a, arr_b, figs=2):
    assert arr_a.shape == arr_b.shape
    assert_array_equal(np.around(arr_a, figs), np.around(arr_b, figs))


def get_date_from_now(days_list):
    return [datetime.now().date() + timedelta(days=days) for days in days_list]


AUS_MIXING_MATRIX = np.array(
    [
        [
            2.60337,
            1.08383,
            0.42239,
            0.27300,
            0.40242,
            0.76062,
            1.14968,
            1.06413,
            0.52875,
            0.30034,
            0.32851,
            0.25771,
            0.18251,
            0.15147,
            0.08213,
            0.04253,
        ],
        [
            0.98614,
            5.81313,
            1.08054,
            0.29122,
            0.18775,
            0.47298,
            0.82713,
            1.04951,
            0.86873,
            0.36094,
            0.22314,
            0.18269,
            0.16513,
            0.12641,
            0.05289,
            0.03981,
        ],
        [
            0.24894,
            1.72576,
            7.62435,
            0.90536,
            0.33994,
            0.29872,
            0.48466,
            0.75606,
            1.07490,
            0.59468,
            0.29195,
            0.14175,
            0.09805,
            0.09860,
            0.06252,
            0.05117,
        ],
        [
            0.14242,
            0.35244,
            2.82577,
            10.39436,
            1.61576,
            0.81015,
            0.64035,
            0.82596,
            1.07949,
            1.08450,
            0.55185,
            0.21646,
            0.08906,
            0.06448,
            0.03204,
            0.02095,
        ],
        [
            0.25851,
            0.20607,
            0.34565,
            2.67611,
            4.40801,
            1.89963,
            1.20463,
            1.09551,
            0.97607,
            1.21057,
            0.83648,
            0.40239,
            0.11051,
            0.05810,
            0.05663,
            0.04365,
        ],
        [
            0.62184,
            0.31654,
            0.24457,
            0.92109,
            2.21208,
            3.73504,
            1.84970,
            1.42727,
            1.21627,
            0.98955,
            0.97164,
            0.48522,
            0.15019,
            0.06809,
            0.03237,
            0.02331,
        ],
        [
            0.84899,
            0.94128,
            0.69315,
            0.51691,
            1.05358,
            1.78707,
            2.90024,
            1.83065,
            1.38650,
            1.06289,
            0.81938,
            0.54919,
            0.21941,
            0.10828,
            0.04871,
            0.04353,
        ],
        [
            0.75985,
            1.11984,
            0.88638,
            0.77692,
            0.75430,
            1.37231,
            1.68074,
            2.92521,
            1.94818,
            1.23281,
            0.88433,
            0.46382,
            0.25541,
            0.17299,
            0.08841,
            0.03382,
        ],
        [
            0.38027,
            0.78820,
            1.10882,
            1.20187,
            0.93225,
            1.21900,
            1.53094,
            1.76698,
            2.72030,
            1.58046,
            1.08058,
            0.39298,
            0.20328,
            0.13483,
            0.08333,
            0.03748,
        ],
        [
            0.33806,
            0.55454,
            0.77220,
            1.62111,
            0.89585,
            0.97944,
            1.18102,
            1.35396,
            1.48261,
            1.96347,
            1.06360,
            0.47029,
            0.15867,
            0.09402,
            0.07397,
            0.06267,
        ],
        [
            0.30804,
            0.66554,
            0.99733,
            1.38147,
            1.05417,
            1.33233,
            1.18105,
            1.17046,
            1.55584,
            1.69563,
            1.84097,
            0.81499,
            0.27351,
            0.13407,
            0.08243,
            0.06900,
        ],
        [
            0.55696,
            0.76363,
            0.76398,
            0.92991,
            0.74030,
            1.18388,
            1.22103,
            0.96533,
            1.08501,
            0.90708,
            1.12601,
            1.42573,
            0.49202,
            0.23027,
            0.09948,
            0.07064,
        ],
        [
            0.52244,
            0.49263,
            0.37283,
            0.51860,
            0.42425,
            0.63332,
            0.74111,
            0.77506,
            0.59676,
            0.47399,
            0.48787,
            0.65517,
            1.07031,
            0.42398,
            0.20537,
            0.08138,
        ],
        [
            0.34454,
            0.52111,
            0.43256,
            0.25258,
            0.31631,
            0.45079,
            0.64835,
            0.65097,
            0.60023,
            0.34647,
            0.38745,
            0.48095,
            0.54333,
            1.05213,
            0.28026,
            0.11303,
        ],
        [
            0.14627,
            0.41869,
            0.39556,
            0.39571,
            0.17624,
            0.29946,
            0.29292,
            0.51662,
            0.57100,
            0.44404,
            0.35645,
            0.29857,
            0.52093,
            0.57059,
            0.77302,
            0.23737,
        ],
        [
            0.25725,
            0.34868,
            0.51622,
            0.41510,
            0.15927,
            0.18122,
            0.27390,
            0.37061,
            0.42511,
            0.42800,
            0.42608,
            0.25936,
            0.18814,
            0.29841,
            0.25747,
            0.41008,
        ],
    ]
)

AUS_WORK_MIXING_MATRIX = np.array(
    [
        [
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00001,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00001,
            0.00000,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.19510,
            0.03313,
            0.03523,
            0.00988,
            0.08004,
            0.02329,
            0.08247,
            0.04881,
            0.01944,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.05906,
            0.79382,
            0.63340,
            0.35306,
            0.32627,
            0.30969,
            0.34804,
            0.27958,
            0.16512,
            0.07364,
            0.00963,
            0.00001,
            0.00000,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.07696,
            0.42691,
            0.82282,
            0.76431,
            0.63235,
            0.71143,
            0.54186,
            0.44218,
            0.33956,
            0.14663,
            0.02640,
            0.00001,
            0.00001,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.09593,
            0.36369,
            0.75895,
            1.23873,
            0.87868,
            0.87600,
            0.81638,
            0.58320,
            0.48214,
            0.20936,
            0.03414,
            0.00002,
            0.00001,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.10775,
            0.19483,
            0.52130,
            0.83306,
            1.08070,
            0.96777,
            0.87490,
            0.72781,
            0.44153,
            0.24213,
            0.03168,
            0.00002,
            0.00000,
            0.00000,
        ],
        [
            0.00000,
            0.00000,
            0.06766,
            0.39681,
            0.44157,
            0.79898,
            0.84812,
            1.19670,
            1.17444,
            0.82810,
            0.60642,
            0.22679,
            0.02293,
            0.00001,
            0.00001,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.07713,
            0.25000,
            0.52430,
            0.79715,
            0.93204,
            1.01850,
            1.25875,
            1.03054,
            0.74494,
            0.24291,
            0.03392,
            0.00001,
            0.00001,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.10485,
            0.31841,
            0.36410,
            0.61057,
            0.77603,
            0.87120,
            0.90921,
            0.90415,
            0.58574,
            0.28013,
            0.02753,
            0.00002,
            0.00001,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.11188,
            0.23504,
            0.29910,
            0.60711,
            0.70152,
            0.74594,
            1.01896,
            1.00145,
            0.76518,
            0.34890,
            0.03247,
            0.00001,
            0.00001,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.16690,
            0.15291,
            0.19510,
            0.34782,
            0.47940,
            0.46156,
            0.59524,
            0.47844,
            0.44611,
            0.26236,
            0.02965,
            0.00001,
            0.00001,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.03252,
            0.01197,
            0.04475,
            0.07731,
            0.08443,
            0.10188,
            0.11113,
            0.11057,
            0.08496,
            0.06164,
            0.00587,
            0.00002,
            0.00001,
            0.00001,
        ],
        [
            0.00001,
            0.00000,
            0.00001,
            0.00002,
            0.00003,
            0.00008,
            0.00007,
            0.00003,
            0.00007,
            0.00006,
            0.00008,
            0.00005,
            0.00005,
            0.00001,
            0.00002,
            0.00001,
        ],
        [
            0.00000,
            0.00000,
            0.00000,
            0.00003,
            0.00002,
            0.00002,
            0.00004,
            0.00004,
            0.00004,
            0.00003,
            0.00002,
            0.00004,
            0.00003,
            0.00002,
            0.00002,
            0.00003,
        ],
        [
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00005,
            0.00005,
            0.00005,
            0.00008,
            0.00003,
            0.00001,
            0.00001,
            0.00000,
            0.00000,
            0.00001,
            0.00000,
            0.00000,
        ],
    ]
)
