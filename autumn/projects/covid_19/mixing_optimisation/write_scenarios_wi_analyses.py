from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from autumn.projects.covid_19.mixing_optimisation.utils import get_wi_scenario_mapping
from autumn.projects.covid_19.mixing_optimisation.write_scenarios import (
    build_optimised_scenario_dictionary,
    drop_all_yml_scenario_files,
    read_decision_vars,
    read_opti_outputs,
)

WI_SCENARIO_MAPPING = get_wi_scenario_mapping(vary_final_mixing=False)


def build_all_wi_scenario_dicts_from_outputs(output_filename):
    opti_outputs_df = read_opti_outputs(output_filename)

    all_wi_sc_params = {}
    for country in OPTI_REGIONS:
        all_wi_sc_params[country] = {}
        for sc_idx, settings in WI_SCENARIO_MAPPING.items():
            decision_vars = read_decision_vars(
                opti_outputs_df,
                country,
                "by_age",
                settings["duration"],
                settings["objective"],
            )
            if decision_vars is not None:
                all_wi_sc_params[country][sc_idx] = build_optimised_scenario_dictionary(
                    country,
                    sc_idx,
                    decision_vars,
                    WI_SCENARIO_MAPPING,
                    for_waning_immunity=True,
                    final_mixing=settings["final_mixing"],
                )
    return all_wi_sc_params


if __name__ == "__main__":
    _all_wi_sc_params = build_all_wi_scenario_dicts_from_outputs("opti_outputs.csv")
    drop_all_yml_scenario_files(_all_wi_sc_params)
