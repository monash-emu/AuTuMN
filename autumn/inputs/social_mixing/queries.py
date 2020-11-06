import numpy as np
from functools import lru_cache

from autumn.inputs.database import get_input_db
from autumn.inputs.demography.queries import get_population_by_agegroup

LOCATIONS = ("all_locations", "home", "other_locations", "school", "work")
MAPPING_ISO_CODE = {
    "MHL": "KIR",
}

# Cache result beecause this gets called 1000s of times during calibration.
@lru_cache(maxsize=None)
def get_country_mixing_matrix(mixing_location: str, country_iso_code: str):
    """
    Load a mixing matrix for a given country and mixing location.
    The rows and columns indices of each matrix represent a 5 year age bracket from 0-80,
    giving us a 16x16 matrix.
    """
    assert mixing_location in LOCATIONS, f"Invalid mixing location {mixing_location}"
    if country_iso_code in MAPPING_ISO_CODE:
        country_iso_code = MAPPING_ISO_CODE[country_iso_code]

    input_db = get_input_db()
    mix_df = input_db.query(
        "social_mixing",
        columns=[f"X{n}" for n in range(1, 17)],
        conditions={
            "iso3": country_iso_code,
            "location": mixing_location,
        },
    )
    matrix = np.array(mix_df)
    assert matrix.shape == (16, 16), "Mixing matrix is not 16x16"
    return matrix


def get_mixing_matrix_specific_agegroups(
    country_iso_code: str, requested_age_breaks: list, time_unit="days"
):
    """
    Build an age-specific mixing matrix using any age categories
    :param country_iso_code: character string of length 3
    :param requested_age_breaks: list of integers
    :param time_unit: one of ['days', 'years']
    :return: numpy matrix
    """
    assert all(
        [age % 5 == 0 for age in requested_age_breaks]
    ), "All breakpoints must be multiples of 5"
    assert time_unit in [
        "days",
        "years",
    ], "The requested time-unit must be either 'days' or 'years'"
    original_matrix = get_country_mixing_matrix("all_locations", country_iso_code)
    original_age_breaks = [i * 5.0 for i in range(16)]
    original_populations = get_population_by_agegroup(
        original_age_breaks, country_iso_code, year=2015  # 2015 to match Prem estimates
    )

    n_req_groups = len(requested_age_breaks)
    # get list of original agegroup indices that match each requested age group index
    agegroup_index_mapping = []
    for i, req_age_break in enumerate(requested_age_breaks):
        mapping_indices = [k for k in range(16) if original_age_breaks[k] >= req_age_break]
        if i < len(requested_age_breaks) - 1:
            mapping_indices = [
                k for k in mapping_indices if original_age_breaks[k] < requested_age_breaks[i + 1]
            ]

        agegroup_index_mapping.append(mapping_indices)

    out_matrix = np.zeros((n_req_groups, n_req_groups))
    for i_contactor in range(n_req_groups):
        total_contactor_group_population = sum(
            [original_populations[h] for h in agegroup_index_mapping[i_contactor]]
        )
        rel_contactor_population_props = [
            original_populations[h] / total_contactor_group_population
            for h in agegroup_index_mapping[i_contactor]
        ]
        for j_contactee in range(n_req_groups):
            total_contacts_new_format = 0.0
            for h, i_contactor_original in enumerate(agegroup_index_mapping[i_contactor]):
                total_contacts_new_format += rel_contactor_population_props[h] * sum(
                    [
                        original_matrix[i_contactor_original, p]
                        for p in agegroup_index_mapping[j_contactee]
                    ]
                )
            out_matrix[i_contactor, j_contactee] = total_contacts_new_format

    if time_unit == "years":
        out_matrix *= 365.25
    return out_matrix
