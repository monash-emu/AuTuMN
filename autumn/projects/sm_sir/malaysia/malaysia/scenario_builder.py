from autumn.models.sm_sir.mixing_matrix.macrodistancing import get_mobility_specific_period

lockdown_title = ["No vaccination",
                  "Vaccine coverage halved",
                  "Nation-wide MCO not implemented",
                  "Recovery MCO not implemented",
                  "Recovery and nation-wide MCOs not implemented",
                  "No vaccination and MCOs"]


def get_all_scenario_dicts(country: str):

    num_scenarios = 1
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"lockdown_1_mobility": {}, "mixing": {}},
            "vaccination": {}
        }

        if i_lockdown_scenario == 0:  # no vaccination
            scenario_dict["vaccination"] = {
                "data_thinning": 4,
                "scenario": 1
            }

        if i_lockdown_scenario == 1:  # vaccine coverage halved
            scenario_dict["vaccination"] = {
                "data_thinning": 4,
                "scenario": 2
            }

        if i_lockdown_scenario == 4:  # What if state and nation-wide MCOs not implemented
            scenario_dict["mobility"]["constant_mobility"] = True
            scenario_dict["mobility"]["scenario_number"] = 4

        if i_lockdown_scenario == 3:  # 13th jan - 31st march MCO not implemented
            scenario_dict["mobility"]["constant_mobility"] = True
            scenario_dict["mobility"]["scenario_number"] = 3

        if i_lockdown_scenario == 2:  # jun 1st - 1st OCt nation wide MCO not implemented
            scenario_dict["mobility"]["constant_mobility"] = True
            scenario_dict["mobility"]["scenario_number"] = 2

        if i_lockdown_scenario == 5:  # What if no vaccination and no MCOs implemented
            scenario_dict["mobility"]["constant_mobility"] = True
            scenario_dict["mobility"]["scenario_number"] = 5

            scenario_dict["vaccination"] = {
                "data_thinning": 4,
                "scenario": 1
            }
        all_scenario_dicts.append(scenario_dict)
    return all_scenario_dicts

