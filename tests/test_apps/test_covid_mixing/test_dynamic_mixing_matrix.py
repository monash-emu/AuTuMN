from datetime import timedelta

import numpy as np
from numpy.testing import assert_allclose

from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.models.covid_19.parameters import Country, Mobility
from autumn.models.sm_sir.mixing_matrix import build_dynamic_mixing_matrix, macrodistancing
from autumn.core.inputs.social_mixing.queries import get_country_mixing_matrix, get_mixing_matrix_specific_agegroups

MM = np.ones([16, 16])
HOME_MM = MM * 0.1
OTHER_LOCATIONS_MM = MM * 0.2
SCHOOL_MM = MM * 0.3
WORK_MM = MM * 0.6

MIXING_MATRICES = {
    'all_locations': MM,
    'home': HOME_MM,
    'other_locations': OTHER_LOCATIONS_MM,
    'school': SCHOOL_MM,
    'work': WORK_MM
}

UNTESTED_PARAMS = {
    "google_mobility_locations": {},
    "smooth_google_data": False,
    "region": None,
}


def test_build_dynamic__with_no_changes():
    """
    Ensure dynamic mixing matrix has no change over time, if no changes are supplied.
    """
    #monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
    mobility_params = {
        "mixing": {},
        "age_mixing": None,
        "microdistancing": {},
        "square_mobility_effect": False,
        **UNTESTED_PARAMS,
    }

    mm_func = build_dynamic_mixing_matrix(
        base_matrices=MIXING_MATRICES, country=Country(iso3="AUS"), mobility=Mobility(**mobility_params)
    )
    mm = mm_func(0)
    assert_allclose(mm, MM, atol=0.01, verbose=True)

    mm = mm_func(111)
    assert_allclose(mm, MM, atol=0.01, verbose=True)

#
# def test_build_dynamic__with_location_mobility_data(monkeypatch):
#     """
#     Ensure dynamic mixing matrix can use location-based mobility data set by the user + Google.
#     """
#
#     def get_fake_mobility_data(*args, **kwargs):
#         vals = {"work": [1, 1.5, 1.3, 1.1]}
#         days = [0, 1, 2, 3]
#         return vals, days
#
#     monkeypatch.setattr(mobility, "get_mobility_data", get_fake_mobility_data)
#     #monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
#     mobility_params = {
#         "mixing": {
#             "school": {
#                 "append": False,
#                 "times": get_date_from_base([0, 1, 2, 3]),
#                 "values": [1, 0.5, 0.3, 0.1],
#             }
#         },
#         "age_mixing": None,
#         "microdistancing": {},
#         "square_mobility_effect": False,
#         **UNTESTED_PARAMS,
#     }
#     mm_func = build_dynamic_mixing_matrix(
#         base_matrices=MIXING_MATRICES,
#         country=Country(iso3="AUS"),
#         mobility=Mobility(**mobility_params),
#     )
#
#     mm = mm_func(0)
#     assert_allclose(mm, MM, atol=0.01, verbose=True)
#
#     mm = mm_func(2)
#     expected_mm = MM.copy() + (0.3 - 1) * SCHOOL_MM + (1.3 - 1) * WORK_MM
#     assert_allclose(mm, expected_mm, atol=0.01, verbose=True)


def test_build_dynamic__with_age_mobility_data():
    """
    Ensure dynamic mixing matrix can use age-based mobility data set by the user.
    """
    mobility_params = {
        "mixing": {},
        "age_mixing": {
            "0": {
                "times": [0, 1, 2, 3],
                "values": [2, 2, 2, 2],
            },
            "5": {
                "times": [0, 1, 2, 3],
                "values": [3, 3, 3, 3],
            },
        },
        "microdistancing": {},
        "square_mobility_effect": False,
        **UNTESTED_PARAMS,
    }
    mm_func = build_dynamic_mixing_matrix(
        base_matrices=MIXING_MATRICES,
        country=Country(iso3="AUS"),
        mobility=Mobility(**mobility_params),
    )
    expected_mm = MM.copy()
    # Add adjustment of 2 to the 1st row and col for the 0-5 age bracket
    expected_mm[0, :] *= 2
    expected_mm[:, 0] *= 2
    # Add adjustment of 3 to the 2nd row and col for the 5-10 age bracket
    expected_mm[1, :] *= 3
    expected_mm[:, 1] *= 3

    mm = mm_func(0)
    assert_allclose(mm, expected_mm, atol=0.01, verbose=True)
    mm = mm_func(2)
    assert_allclose(mm, expected_mm, atol=0.01, verbose=True)


def test_build_dynamic__with_microdistancing():
    """
    Ensure dynamic mixing matrix can use microdistancing set by the user.
    """
    mobility_params = {
        "mixing": {},
        "age_mixing": None,
        "microdistancing": {
            "foo": {
                "function_type": "empiric",
                "parameters": {
                    "max_effect": 0.6,
                    "times": [0, 365],
                    "values": [0, 0.1],
                },
                "locations": ["work"],
            }
        },
        "square_mobility_effect": False,
        **UNTESTED_PARAMS,
    }
    mm_func = build_dynamic_mixing_matrix(
        base_matrices=MIXING_MATRICES,
        country=Country(iso3="AUS"),
        mobility=Mobility(**mobility_params),
    )
    mm = mm_func(0)
    assert_allclose(mm, MM, atol=0.01, verbose=True)

    mm = mm_func(365)
    expected_mm = MM.copy() + ((1 - 0.06) - 1) * WORK_MM
    assert_allclose(mm, expected_mm, atol=0.01, verbose=True)


# def test_build_dynamic__with_everything(monkeypatch):
#     """
#     Ensure dynamic mixing matrix can use:
#         - microdistancing set by the user
#         - user provided mobility data
#         - google mobility data
#         - age based mixing
#     """
#
#     def get_fake_mobility_data(*args, **kwargs):
#         vals = {"work": [1, 1.5, 1.3, 1.1]}
#         days = [0, 1, 2, 3]
#         return vals, days
#
#     monkeypatch.setattr(mobility, "get_mobility_data", get_fake_mobility_data)
#     #monkeypatch.setattr(location_adjuster, "get_country_mixing_matrix", _get_country_mixing_matrix)
#     mobility_params = {
#         "mixing": {
#             "school": {
#                 "append": False,
#                 "times": get_date_from_base([0, 1, 2, 3]),
#                 "values": [1, 0.5, 0.3, 0.1],
#             }
#         },
#         "age_mixing": {
#             "0": {
#                 "times": [0, 1, 2, 3],
#                 "values": [0.5, 0.5, 0.5, 0.5],
#             },
#             "5": {
#                 "times": [0, 1, 2, 3],
#                 "values": [0.3, 0.3, 0.3, 0.3],
#             },
#         },
#         "microdistancing": {
#             "foo": {
#                 "function_type": "empiric",
#                 "parameters": {
#                     "max_effect": 0.6,
#                     "times": [0, 1, 2, 3],
#                     "values": [0, 0.1, 0.2, 0.3],
#                 },
#                 "locations": ["work"],
#             }
#         },
#         "square_mobility_effect": True,
#         **UNTESTED_PARAMS,
#     }
#     mm_func = build_dynamic_mixing_matrix(
#         base_matrices=MIXING_MATRICES,
#         country=Country(iso3="AUS"),
#         mobility=Mobility(**mobility_params),
#     )
#     # Expect only age-based mixing to occur.
#     mm = mm_func(0)
#     expected_mm = MM.copy()
#     # Add adjustment of 2 to the 1st row and col for the 0-5 age bracket
#     expected_mm[0, :] *= 0.5
#     expected_mm[:, 0] *= 0.5
#     # Add adjustment of 3 to the 2nd row and col for the 5-10 age bracket
#     expected_mm[1, :] *= 0.3
#     expected_mm[:, 1] *= 0.3
#     assert_allclose(mm, expected_mm, atol=0.01, verbose=True)
#
#     # Expect age based, microdistancing, google mobility and user-specified mobility to be used.
#     mm = mm_func(3)
#     work_microdistancing = (1 - 0.3 * 0.6) ** 2
#     work_google_mobility = 1.1 ** 2
#     work_factor = work_microdistancing * work_google_mobility - 1
#     school_factor = 0.1 ** 2 - 1
#     expected_mm = MM.copy() + work_factor * WORK_MM + school_factor * SCHOOL_MM
#     # Add adjustment of 2 to the 1st row and col for the 0-5 age bracket
#     expected_mm[0, :] *= 0.5
#     expected_mm[:, 0] *= 0.5
#     # Add adjustment of 3 to the 2nd row and col for the 5-10 age bracket
#     expected_mm[1, :] *= 0.3
#     expected_mm[:, 1] *= 0.3
#     assert_allclose(mm, expected_mm, atol=0.01, verbose=True)


def get_date_from_base(days_list):
    return [COVID_BASE_DATETIME + timedelta(days=days) for days in days_list]


def _get_country_mixing_matrix(sheet_type, iso3):
    if sheet_type == "home":
        return HOME_MM
    if sheet_type == "other_locations":
        return OTHER_LOCATIONS_MM
    if sheet_type == "school":
        return SCHOOL_MM
    if sheet_type == "work":
        return WORK_MM
    else:
        return MM


def test_age_mixing_matrix_variable_agegroups__smoke_test():
    requested_age_breaks = [0, 20, 50]
    out_matrix = get_mixing_matrix_specific_agegroups("AUS", requested_age_breaks)
    assert out_matrix.shape == (3, 3)


def test_age_mixing_matrix_variable_agegroups__conservation():
    requested_age_breaks = [i * 5.0 for i in range(16)]  # same as original Prem age groups
    prem_matrix = get_country_mixing_matrix("all_locations", "AUS")
    out_matrix = get_mixing_matrix_specific_agegroups("AUS", requested_age_breaks)
    assert np.array_equal(out_matrix, prem_matrix)
