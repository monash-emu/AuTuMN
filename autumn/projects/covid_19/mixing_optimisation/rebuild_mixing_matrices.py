from autumn.projects.covid_19.mixing_optimisation.constants import PHASE_2_START_TIME
from autumn.models.covid_19.mixing_matrix import (
    build_dynamic_mixing_matrix,
)
from autumn.tools.inputs.demography.queries import get_iso3_from_country_name

from .mixing_opti import build_params_for_phases_2_and_3


# FIXME this is broken
def get_mixing_matrices(
    output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False
):

    iso_3 = get_iso3_from_country_name(country.title()) if country != "united-kingdom" else "GBR"
    params, decision_vars = get_mle_params_and_vars(
        output_dir, country, config, mode, objective, from_streamlit
    )

    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_vars[0],
            "school": decision_vars[1],
            "work": decision_vars[2],
        }
        decision_vars = new_decision_variables

    sc_1_params = build_params_for_phases_2_and_3(decision_vars, config, mode)
    if mode == "by_location":
        sc_1_params["mixing_age_adjust"] = {}

    # FIXME: this is probably broken!
    mixing_func = build_dynamic_mixing_matrix(
        iso_3,
        country,
        mixing=sc_1_params["mixing"],
        mixing_age_adjust=sc_1_params["mixing_age_adjust"],
        npi_effectiveness_params={},
        google_mobility_locations={
            "work": {"workplaces": 1.},
            "other_locations": {
                "retail_and_recreation": 0.25,
                "grocery_and_pharmacy": 0.25,
                "transit_stations": 0.25,
            },
            "home": {"residential": 1.}
        },
        is_periodic_intervention=False,
        periodic_int_params={},
        periodic_end_time=0.0,
        microdistancing_params={},
        smooth_google_data=True,
    )

    original_prem = mixing_func(10000.0)
    optimised = mixing_func(PHASE_2_START_TIME + 10.0)

    return original_prem, optimised
