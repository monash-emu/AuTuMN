from autumn.models.covid_19.mixing_matrix.macrodistancing import get_mobility_specific_period

scenario_start_time = [505, 495, 556, 481, 506, 640]  # 505 - 20 May, 476 - 29th April,
# 567 - 21st July 2021, 481 - 25 April, 507- May 21, 650 - Oct 01
lockdown_title = ["No lockdowns placed", "What if lockdown was initiated from May 9 - June 9",
                  "What if lockdown was initiated from July 20 - Aug 30", "No vaccination",
                  "Second lockdown based on WHO mortalilty threshold",
                  "Faster increase in mobility after lockdown ends on 01st October"]
scenario_end_time = [791, 791, 791, 791, 791, 791]


def get_vaccine_roll_out(lockdown_scenario):
    age_mins = [15, 15]
    roll_out_components = []
    start_time = [481, 655]
    end_time = [655, 732]

    for i_age_min, age_min in enumerate(age_mins):

        if lockdown_scenario == 3:
            coverage = [0.00001, 0.00001]  # no vaccine scenario
        else:
            coverage = [0.7587, 0.18]  # vaccine scenarios

        component = {"supply_period_coverage":
                         {"coverage": coverage[i_age_min], "start_time": start_time[i_age_min], "end_time": end_time[i_age_min]},}
        component["age_min"] = age_min
        roll_out_components.append(component)
    return roll_out_components


def get_all_scenario_dicts(country: str):
    num_scenarios = 5
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "time": {"start": scenario_start_time[i_lockdown_scenario], "end": scenario_end_time[i_lockdown_scenario]},
            "description": "scenario:"+f"{i_lockdown_scenario+1}" + f" {lockdown_title[i_lockdown_scenario]}",
            "mobility": {"mixing": {}},
            "vaccination": {"roll_out_components": []}
        }
        # mobility parameters
        if i_lockdown_scenario == 0:
            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + [scenario_start_time[i_lockdown_scenario]+1],
                    "values": [["repeat_prev"]] + [1.5]
                }

        if i_lockdown_scenario == 1:  # What if lockdown was initiated from May 9 - June 9

            # lockdown mobility from 21May -21 June applied from May 9 - June 9
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [507, 537])

            times1 = [*range(496, 526)]  # lockdown values from May 21 - June 21 is assigned from May 9 -June 9

            # from May 30 - 21 June, the average mobility after lockdown
            times2 = [*range(526, 539)]
            values2 = {'work': [0.6] * len(times2), 'other_locations': [0.72] * len(times2)}

            # From June 22nd onwards actual mobility levels
            times3, values3 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [539, 670])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times2 +
                              times3 + [times3[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] +
                              values3[key_loc] + [["repeat_prev"]]
                }
        if i_lockdown_scenario == 2:  # What if lockdown was initiated from July 21 - Aug 31

            # lockdown mobility from 21Aug -01 Oct is assigned from July 21 - Aug 31 in the scenario
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [599, 630])
            times1 = [*range(568, 599)]  # July 21 - Aug 31

            times3 = [*range(599, 641)]
            values3 = {'work': [0.5] * len(times3), 'other_locations': [0.5] * len(times3)}

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown) frm Sep 01 st onwards
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [641, 670])
            #times2 = [*range(610, 639)]  # from Sep 01st onwards

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times3 +
                             times2 + [times2[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values3[key_loc] + values2[key_loc] +
                              [["repeat_prev"]]
                }
        if i_lockdown_scenario == 4:  # What if lockdown was initiated based on death indicator on 13th June 2021            # lockdown mobility from 21May -21 June applied from April 29 - May 29
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [507, 537])

            times1 = [*range(530, 560)]  # lockdown values from May 21 - June 21 is assigned from 13June -July 13th

            # from May 21 - 13 June, the average mobility before lockdown
            times2 = [*range(507, 530)]
            values2 = {'work': [0.65] * len(times2), 'other_locations': [0.8] * len(times2)}

            # From July 13th onwards actual mobility levels
            times3, values3 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [560, 670])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times2 + times1 +
                              times3 + [times3[-1] + 1],
                    "values": [["repeat_prev"]] + values2[key_loc] + values1[key_loc] +
                              values3[key_loc] + [["repeat_prev"]]
                }

        if i_lockdown_scenario == 5:  # "Faster increase in mobility after lockdown ends on 01st October"
            for key_loc in ["other_locations", "work"]:
                if key_loc == "other_locations":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1],
                        "values": [["repeat_prev"]] + [1.04]
                    }
                if key_loc == "work":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1],
                        "values": [["repeat_prev"]] + [1.0]
                    }

        # vaccination parameters
        scenario_dict["vaccination"]["roll_out_components"] = get_vaccine_roll_out(i_lockdown_scenario)
        if i_lockdown_scenario == 3:
            scenario_dict["vaccination"]["standard_supply"] = False

        # school openings
        if i_lockdown_scenario == 4:

            scenario_dict["mobility"]["mixing"]["school"] = {
                "append": False,
                "times": [653, 654],
                "values": [0.0, 0.4]
            }
        elif i_lockdown_scenario == 5:
            scenario_dict["mobility"]["mixing"]["school"] = {
                "append": False,
                "times": [653, 654],
                "values": [0.0, 1.0]
            }
        else:
            scenario_dict["mobility"]["mixing"]["school"] = {
                "append": False,
                "times": [653, 654],
                "values": [0.0, 0.0]
            }


        all_scenario_dicts.append(scenario_dict)
    return all_scenario_dicts


