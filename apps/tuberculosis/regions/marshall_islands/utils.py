import yaml


INTERVENTION_RATE = {
    'time_variant_acf': 3.22,
    'time_variant_ltbi_screening': 3.22
}


def define_all_scenarios(periodic_frequencies=[2, 5]):
    scenario_details = {}
    sc_idx = 0

    """
    Baseline: the current situation
    """
    scenario_details[sc_idx] = {"sc_title": 'Status quo'}

    """
    Scenario 1: removing Majuro and Ebeye interventions
    """
    sc_idx += 1
    scenario_details[sc_idx] = {"sc_title": 'No interventions'}
    scenario_details[sc_idx]['params'] = {
        'time_variant_acf': [],
        'time_variant_ltbi_screening': []
    }

    """
    Scenario 2: removing the LTBI component from the existing interventions
    """
    sc_idx += 1
    scenario_details[sc_idx] = {"sc_title": 'No LTBI screening'}
    scenario_details[sc_idx]['params'] = {
        'time_variant_ltbi_screening': []
    }

    """
    Periodic ACF scenarios
    """
    for frequency in periodic_frequencies:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": f"ACF every {frequency} years"}
        scenario_details[sc_idx]["params"] = get_periodic_sc_params(frequency, type='ACF')

    """
    Periodic ACF + LTBI screening scenarios
    """
    for frequency in periodic_frequencies:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": f"ACF and LTBI every {frequency} years"}
        scenario_details[sc_idx]["params"] = get_periodic_sc_params(frequency, type='ACF_LTBI')

    """
    Sensitivity analysis around future diabetes prevalence
    """
    diabetes_scenarios = (
        ('Declining diabetes prevalence', 0.8),
        ('Increasing diabetes prevalence', 1.2)
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
    scenario_details[sc_idx] = {"sc_title": 'Intensive PT in household contacts'}
    scenario_details[sc_idx]['params'] = {
        "hh_contacts_pt": {
            "start_time": 2020,
            "prop_smearpos_among_prev_tb": 1.,
            "prop_hh_transmission": .30,
            "prop_hh_contacts_screened": 1.,
            "prop_pt_completion": 1.,
        }
    }

    return scenario_details


def get_periodic_sc_params(frequency, type='ACF'):
    """
    :param frequency: period at which interventions are implemented
    :param type: one of ['ACF', 'ACF_LTBI']
    :return:
    """
    params = {}

    # include existing interventions
    params['time_variant_acf'] = [
        {
            'stratum_filter': {"location": "ebeye"},
            'time_variant_screening_rate': {
                2017: 0., 2017.01: 3.79, 2017.5: 3.79, 2017.51: 0.
            }
        },
        {
            'stratum_filter': {"location": "majuro"},
            'time_variant_screening_rate': {
                2018: 0., 2018.01: 3.22, 2018.5: 3.22, 2018.51: 0.
            }
        }
    ]
    params['time_variant_ltbi_screening'] = [
        {
            'stratum_filter': {"location": "majuro"},
            'time_variant_screening_rate': {
                2018: 0., 2018.01: 3.22, 2018.5: 3.22, 2018.51: 0.
            }
        }
    ]

    interventions_to_add = ["time_variant_acf", "time_variant_ltbi_screening"] if type == 'ACF_LTBI' else\
        ["time_variant_acf"]

    for intervention in interventions_to_add:
        for location in ["majuro", "ebeye", "other"]:
            is_location_implemented = False
            for local_intervention in params[intervention]:
                if local_intervention['stratum_filter']['location'] == location:
                    is_location_implemented = True
                    local_intervention['time_variant_screening_rate'].update(
                         make_periodic_time_series(INTERVENTION_RATE[intervention], frequency)
                    )
                    break
            if not is_location_implemented:
                params[intervention].append(
                    {
                        'stratum_filter': {"location": location},
                        'time_variant_screening_rate': make_periodic_time_series(INTERVENTION_RATE[intervention], frequency)
                    }
                )

    return params


def make_periodic_time_series(rate, frequency):
    time_series = {}
    year = 2021
    while year < 2050:
        time_series[year] = 0.
        time_series[year + .01] = rate
        time_series[year + .5] = rate
        time_series[year + .51] = 0.
        year += frequency

    return time_series


def drop_all_yml_scenario_files(all_sc_params):
    yaml.Dumper.ignore_aliases = lambda *args: True
    for sc_idx, sc_details in all_sc_params.items():
        if sc_idx == 0 or "params" not in sc_details:
            continue

        params_to_dump = sc_details['params']
        params_to_dump['parent'] = 'apps/tuberculosis/regions/marshall_islands/params/default.yml'
        params_to_dump['time'] = {'start': 2016, 'critical_ranges': [[2017., 2049.]]}

        param_file_path = f"params/scenario-{sc_idx}.yml"
        with open(param_file_path, "w") as f:
            yaml.dump(params_to_dump, f)


if __name__ == "__main__":
    all_scenarios = define_all_scenarios(periodic_frequencies=[2, 5])
    drop_all_yml_scenario_files(all_scenarios)

