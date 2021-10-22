from autumn.models.covid_19.mixing_matrix.mobility import get_mobility_specific_period

scenario_start_time = [505, 476, 556]  # 505 - 20 May, 476 - 21st April,  556 - 9th July 2021
lockdown_title = ["No lockdowns placed", "What if lockdown was initiated from April 21 - June 21",
                  "What if lockdown was initiated from July 10 - Oct 01"]


def get_all_scenario_dicts(country: str):
    num_scenarios = 3
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "time": {"start": scenario_start_time[i_lockdown_scenario]},
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"mixing": {}},
        }
        # mobility parameters
        if i_lockdown_scenario == 0:
            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + [scenario_start_time[i_lockdown_scenario]+1],
                    "values": [["repeat_prev"]] + [1.1]
                }

        if i_lockdown_scenario == 1:  # What if lockdown was initiated from April 21 - June 21

            # lockdown mobility from 21May -21 June
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': ['workplaces'],
                                                            'other_locations': ['retail_and_recreation',
                                                                                'grocery_and_pharmacy',
                                                                                'transit_stations'],
                                                            'home': ['residential']}, [507, 537])

            times1 = [*range(477, 507)]  # lockdown values from May 21 - June 21 is assigned from April 21 -May 20

            # from May 20 - 21 June, the average mobility from from May 21 - June 21
            times5 = [*range(507, 539)]
            values5 = {'work': [0.4] * len(times5), 'other_locations': [0.4] * len(times5)}

            # mobility from 22 June -21st August
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': ['workplaces'],
                                                            'other_locations': ['retail_and_recreation',
                                                                                'grocery_and_pharmacy',
                                                                                'transit_stations'],
                                                            'home': ['residential']}, [539, 599])

            # In the scenario from August 21 - Oct 01 applying average mobility observed after lockdown
            # workplaces: 0.7 and other locations 0.9
            times3 = [*range(599, 641)]
            values3 = {'work': [0.6] * len(times3), 'other_locations': [0.8] * len(times3)}

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown)
            times4, values4 = get_mobility_specific_period(country, None,
                                                           {'work': ['workplaces'],
                                                            'other_locations': ['retail_and_recreation',
                                                                                'grocery_and_pharmacy',
                                                                                'transit_stations'],
                                                            'home': ['residential']}, [641, 645])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times5 +
                             times2 + times3 + times4 + [times4[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values5[key_loc] +
                              values2[key_loc] + values3[key_loc] + values4[key_loc] + [["repeat_prev"]]
                }
        if i_lockdown_scenario == 2:  # What if lockdown was initiated from July 10 - Oct 01
            # lockdown mobility from 21Aug -01 Oct
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': ['workplaces'],
                                                            'other_locations': ['retail_and_recreation',
                                                                                'grocery_and_pharmacy',
                                                                                'transit_stations'],
                                                            'home': ['residential']}, [599, 641])

            # lockdown values from 21Aug -01 Oct is assigned from July 10 - Aug 20 in the scenario
            times1 = [*range(557, 599)]

            # In the scenario, from Aug 21 - Oct 01 applying average mobility observed during lockdown
            # workplaces: 0.4 and other locations 0.4
            times3 = [*range(599, 641)]
            values3 = {'work': [0.3] * len(times3), 'other_locations': [0.3] * len(times3)}

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown)
            times4, values4 = get_mobility_specific_period(country, None,
                                                           {'work': ['workplaces'],
                                                            'other_locations': ['retail_and_recreation',
                                                                                'grocery_and_pharmacy',
                                                                                'transit_stations'],
                                                            'home': ['residential']}, [641, 645])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 +
                             times3 + times4 + [times4[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values3[key_loc] +
                              values4[key_loc] + [["repeat_prev"]]
                }
        all_scenario_dicts.append(scenario_dict)


    return all_scenario_dicts