from autumn.models.sm_sir.mixing_matrix.macrodistancing import get_mobility_specific_period

lockdown_title = ["No vaccination",
                  "Vaccine coverage halved",
                  "What if state and nation-wide MCOs not implemented",
                  "What if nation-wide MCO not implemented"]


def get_all_scenario_dicts(country: str):

    num_scenarios = 4
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"mixing": {}},
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

        if i_lockdown_scenario == 2:  # What if state and nation-wide MCOs not implemented

            # from Jan 13-March 31st other location mobility and work are fixed at respectively, 0.95 and 0.90
            times1 = [*range(379, 457)]
            values1 = {'work': [0.95] * len(times1), 'other_locations': [0.9] * len(times1)}

            # actual mobility from 31st March to 1st June
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [457, 518])

            # from June 1-28th other location mobility and work are fixed at 0.85
            times3 = [*range(518, 546)]
            values3 = {'work': [0.85] * len(times3), 'other_locations': [0.85] * len(times3)}

            # mobility from 29th of June onwards
            times4, values4 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [546, 900])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [378] + times1+ times2 + times3 + times4 + [times4[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc]+ values2[key_loc] + values3[key_loc] +
                              values4[key_loc] + [["repeat_prev"]],
                }

        if i_lockdown_scenario == 3:  # jun 1st - 28th june MCO not implemented

            # from June 1-28th other location mobility and work are fixed at 0.85
            times1 = [*range(518, 545)]
            values1 = {'work': [0.85] * len(times1), 'other_locations': [0.85] * len(times1)}

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

