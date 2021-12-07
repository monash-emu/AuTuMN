from autumn.models.covid_19.mixing_matrix.macrodistancing import get_mobility_specific_period

scenario_start_time = [481, 481, 518]
scenario_end_time = [670, 670, 670]  # 31st October , 2021
lockdown_title = ["No vaccination", "Vaccine low coverage",
                  "No lockdown from 1-28th June"]


def get_vaccine_roll_out(lockdown_scenario):
    age_mins = [10, 10]
    roll_out_components = []
    start_time = [481, 678]
    end_time = [678, 732]

    for i_age_min, age_min in enumerate(age_mins):

        if lockdown_scenario == 0:
            coverage = [0.00001, 0.00001]  # no vaccine scenario
        elif lockdown_scenario == 1:
            coverage = [0.413, 0.9054]  # low vaccine scenario
        else:
            coverage = [0.89, 0.52]  # vaccine scenarios

        component = {"supply_period_coverage":
                         {"coverage": coverage[i_age_min], "start_time": start_time[i_age_min], "end_time": end_time[i_age_min]},}
        component["age_min"] = age_min
        roll_out_components.append(component)
    return roll_out_components


def get_all_scenario_dicts(country: str):
    num_scenarios = 3
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "time": {"start": scenario_start_time[i_lockdown_scenario], "end": scenario_end_time[i_lockdown_scenario]},
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"mixing": {}},
            "vaccination": {"roll_out_components": []}
        }
        # mobility parameters
        if i_lockdown_scenario == 2:  # No lockdown from 1-28th June, 2021
            # no lockdown  from 01 -28 June, 2021
            times1 = [*range(519, 545)]
            values1 = {'work': [1.0] * len(times1), 'other_locations': [1.0] * len(times1)}

            # mobility from 29 June -31st Oct
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [545, 600])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times2 + [times2[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] + [["repeat_prev"]]
                }

        # vaccination parameters
        scenario_dict["vaccination"]["roll_out_components"] = get_vaccine_roll_out(i_lockdown_scenario)

        all_scenario_dicts.append(scenario_dict)

    return all_scenario_dicts

