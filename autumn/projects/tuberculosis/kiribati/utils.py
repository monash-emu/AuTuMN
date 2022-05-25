import yaml

INTERVENTION_RATE = {"time_variant_acf": 1.66, "time_variant_ltbi_screening": .9}

BASELINE_POST_INTERVENTION_RATE = {"time_variant_acf": 0.0, "time_variant_ltbi_screening": 0.0}

# international immigration: 1,434 between April 2006 and March 2011 (5 years)
N_IMMIGRANTS = 300  # per year
SA_PARAM_VALUES = {
    "sa_importation": [0, .1, .20, .30, .40],  # proportion of immigrants infected with LTBI
    "sa_screening": [0.5, .6, .7, .8, .9],
}


def define_all_scenarios(periodic_frequencies=[2, 5, 10]):
    scenario_details = {}
    sc_idx = 0

    """
    Baseline: the current situation
    """
    scenario_details[sc_idx] = {"sc_title": "Status quo"}

    """
    Scenario 1: removing Tarawa and Other interventions
    """
    sc_idx += 1
    scenario_details[sc_idx] = {"sc_title": "No interventions"}
    scenario_details[sc_idx]["params"] = {
        "time_variant_acf": [],
        "time_variant_ltbi_screening": [],
        "awareness_raising": {
            "relative_screening_rate": 1.,
            "scale_up_range": [3000, 3001]
        },
    }

    """
    Scenario 2: removing the LTBI component from the existing interventions
    """
    sc_idx += 1
    scenario_details[sc_idx] = {"sc_title": "No LTBI screening"}
    scenario_details[sc_idx]["params"] = {"time_variant_ltbi_screening": []}

    """
    Periodic ACF scenarios
    """
    for frequency in periodic_frequencies:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": f"ACF every {frequency} years"}
        scenario_details[sc_idx]["params"] = get_periodic_sc_params(frequency, type="ACF")

    """
    Periodic ACF + LTBI screening scenarios
    """
    for frequency in periodic_frequencies:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": f"ACF and LTBI every {frequency} years"}
        scenario_details[sc_idx]["params"] = get_periodic_sc_params(frequency, type="ACF_LTBI")

    """
    Sensitivity analysis around future diabetes prevalence
    """
    diabetes_scenarios = (
        ("Declining diabetes prevalence", 0.8),
        ("Increasing diabetes prevalence", 1.2),
    )
    for diabetes_sc in diabetes_scenarios:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": diabetes_sc[0]}
        scenario_details[sc_idx]["params"] = {
            "extra_params": {"future_diabetes_multiplier": diabetes_sc[1]}
        }

    """
    Extremely high coverage and performance of PT in household contacts
    """
    sc_idx += 1
    scenario_details[sc_idx] = {"sc_title": "Intensive PT in household contacts"}
    scenario_details[sc_idx]["params"] = {
        "hh_contacts_pt": {
            "start_time": 2020,
            "prop_smearpos_among_prev_tb": 1.0,
            "prop_hh_transmission": 0.30,
            "prop_hh_contacts_screened": 1.0,
            "prop_pt_completion": 1.0,
        }
    }

    return scenario_details


def get_periodic_sc_params(frequency, type="ACF"):
    """
    :param frequency: period at which interventions are implemented
    :param type: one of ['ACF', 'ACF_LTBI']
    :return:
    """
    params = {}

    # include existing interventions
    params["time_variant_acf"] = [
        {
            "stratum_filter": {"location": "starawa"},
            "time_variant_screening_rate": {2017: 0.0, 2017.01: 1.9, 2018: 1.9, 2018.01: 0.0},
        },
        {
            "stratum_filter": {"location": "other"},
            "time_variant_screening_rate": {2018: 0.0, 2018.01: INTERVENTION_RATE["time_variant_acf"], 2019: INTERVENTION_RATE["time_variant_acf"], 2019.01: 0.0},
        },
    ]
    params["time_variant_ltbi_screening"] = [
        {
            "stratum_filter": {"location": "other"},
            "time_variant_screening_rate": {
                2018: 0.0,
                2018.01: INTERVENTION_RATE["time_variant_ltbi_screening"],
                2019: INTERVENTION_RATE["time_variant_ltbi_screening"],
                2019.01: BASELINE_POST_INTERVENTION_RATE["time_variant_ltbi_screening"],
            },
        },
        # {
        #     "stratum_filter": {"location": "ebeye"},
        #     "time_variant_screening_rate": {
        #         2018.01: 0.0,
        #         2019: BASELINE_POST_INTERVENTION_RATE["time_variant_ltbi_screening"],
        #     },
        # },
        # {
        #     "stratum_filter": {"location": "other"},
        #     "time_variant_screening_rate": {
        #         2018.01: 0.0,
        #         2019: BASELINE_POST_INTERVENTION_RATE["time_variant_ltbi_screening"],
        #     },
        # },
    ]

    interventions_to_add = (
        ["time_variant_acf", "time_variant_ltbi_screening"]
        if type == "ACF_LTBI"
        else ["time_variant_acf"]
    )

    for intervention in interventions_to_add:
        for location in ["starawa", "other"]:
            is_location_implemented = False
            for local_intervention in params[intervention]:
                if local_intervention["stratum_filter"]["location"] == location:
                    is_location_implemented = True
                    local_intervention["time_variant_screening_rate"].update(
                        make_periodic_time_series(
                            INTERVENTION_RATE[intervention],
                            frequency,
                            BASELINE_POST_INTERVENTION_RATE[intervention],
                        )
                    )
                    break
            if not is_location_implemented:
                params[intervention].append(
                    {
                        "stratum_filter": {"location": location},
                        "time_variant_screening_rate": make_periodic_time_series(
                            INTERVENTION_RATE[intervention],
                            frequency,
                            BASELINE_POST_INTERVENTION_RATE[intervention],
                        ),
                    }
                )

    return params


def make_periodic_time_series(rate, frequency, baseline_rate=0.0):
    time_series = {}
    year = 2021
    while year < 2050:
        time_series[year] = baseline_rate
        time_series[year + 0.01] = rate
        time_series[year + 1.0] = rate
        time_series[year + 1.01] = baseline_rate
        year += frequency

    return time_series


def drop_all_yml_scenario_files(all_sc_params):
    yaml.Dumper.ignore_aliases = lambda *args: True
    for sc_idx, sc_details in all_sc_params.items():
        if sc_idx == 0 or "params" not in sc_details:
            continue

        params_to_dump = sc_details["params"]
        params_to_dump["time"] = {"start": 2016}

        param_file_path = f"params/scenario-{sc_idx}.yml"
        with open(param_file_path, "w") as f:
            yaml.dump(params_to_dump, f)


# SA scenarios
def make_sa_scenario_list(sa_type):
    sa_scenarios = []
    for v in SA_PARAM_VALUES[sa_type]:
        if sa_type == "sa_importation":
            sc_dict = {
                "time": {'start': 2015},
                'import_ltbi_cases': {
                    "start_time": 2016,
                    "n_cases_per_year": v * N_IMMIGRANTS,
                }
            }
        elif sa_type == "sa_screening":
            sc_dict = {
                "time": {'start': 2015},
                'ltbi_screening_sensitivity': v,
            }

        sa_scenarios.append(
            sc_dict
        )

    return sa_scenarios


if __name__ == "__main__":
    all_scenarios = define_all_scenarios(periodic_frequencies=[2, 5, 10])
    drop_all_yml_scenario_files(all_scenarios)
