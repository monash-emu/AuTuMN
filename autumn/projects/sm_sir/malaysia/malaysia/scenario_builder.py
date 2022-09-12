from autumn.models.sm_sir.mixing_matrix.macrodistancing import get_mobility_specific_period

lockdown_title = ["What if nation-wide lockdown was not implemented"]

def get_all_scenario_dicts(country: str):
    num_scenarios = 1
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"mixing": {}},
        }

        if i_lockdown_scenario == 0:

            # from June 1-28th other location mobility and work are fixed at respectively, 0.69 and 0.70
            times1 = [*range(518, 545)]
            values1 = {'work': [0.7] * len(times1), 'other_locations': [0.69] * len(times1)}

            # mobility from 29th of June onwards
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [545, 900])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [517] + times1 + times2 +[times2[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] + [["repeat_prev"]],
                }

        all_scenario_dicts.append(scenario_dict)
    return all_scenario_dicts

