import yaml


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

    """
    Periodic ACF + LTBI screening scenarios
    """
    for frequency in periodic_frequencies:
        sc_idx += 1
        scenario_details[sc_idx] = {"sc_title": f"ACF and LTBI every {frequency} years"}

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

    return scenario_details


def drop_all_yml_scenario_files(all_sc_params):
    yaml.Dumper.ignore_aliases = lambda *args: True
    for sc_idx, sc_details in all_sc_params.items():
        if sc_idx == 0 or "params" not in sc_details:
            continue

        params_to_dump = sc_details['params']
        params_to_dump['parent'] = 'apps/tuberculosis/regions/marshall_islands/params/default.yml'
        params_to_dump['time'] = {'start': 2016}

        param_file_path = f"params/scenario-{sc_idx}.yml"
        with open(param_file_path, "w") as f:
            yaml.dump(params_to_dump, f)


if __name__ == "__main__":
    all_scenarios = define_all_scenarios(periodic_frequencies=[2, 5])
    drop_all_yml_scenario_files(all_scenarios)
