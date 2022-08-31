from autumn.models.covid_19.mixing_matrix.macrodistancing import get_mobility_specific_period

scenario_start_time = [505, 578, 481]  # 505 - 20 May, 578 - 1 August, 481 - 25 April
lockdown_title = ["No lockdowns placed",
                  "What if lockdown was initiated from Aug 1 - Sep 11",
                  "No vaccination"]
scenario_end_time = [791, 791, 791]


def get_vaccine_roll_out(lockdown_scenario):
    age_mins = [15, 15]
    roll_out_components = []
    start_time = [481, 655]
    end_time = [655, 732]

    for i_age_min, age_min in enumerate(age_mins):

        if lockdown_scenario == 2:
            coverage = [0.00001, 0.00001]  # no vaccine scenario
        else:
            coverage = [0.7587, 0.18]  # vaccine scenarios

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
        if i_lockdown_scenario == 0: # no lockdowns implemented

            # from May 21 - 21 June, the average mobility observed before May 21
            times1 = [*range(507, 539)]
            values1 = {'work': [0.72] * len(times1), 'other_locations': [0.84] * len(times1)}

            # from August 21 - 01st October the average mobility observed before August 21
            times2 = [*range(599, 641)]
            values2 = {'work': [0.87] * len(times2), 'other_locations': [1.1] * len(times2)}

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times2 + [times2[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] +
                              [["repeat_prev"]]
                }

        if i_lockdown_scenario == 1:  # What if lockdown was initiated from Aug 1 - Sep 11

            # lockdown mobility from 21Aug -01 Oct is assigned from Aug 1 - Sep 11 in the scenario
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [599, 640])
            times1 = [*range(579, 620)]  # Aug 1 - 11 Sep

            times2 = [*range(620, 640)]
            values2 = {'work': [0.71] * len(times2), 'other_locations': [0.96] * len(times2)}

            times3, values3 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [640, 670])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times2 +
                    times3 + [times3[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] + values3[key_loc] +
                              [["repeat_prev"]]
                }

        # scenario 3 is the same mobility as baseline but no vaccination
        # vaccination parameters
        scenario_dict["vaccination"]["roll_out_components"] = get_vaccine_roll_out(i_lockdown_scenario)
        if i_lockdown_scenario == 2:
            scenario_dict["vaccination"]["standard_supply"] = False
        else:
            scenario_dict["vaccination"]["standard_supply"] = True

        all_scenario_dicts.append(scenario_dict)
    return all_scenario_dicts


