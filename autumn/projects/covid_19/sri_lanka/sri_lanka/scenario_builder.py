from autumn.models.covid_19.mixing_matrix.macrodistancing import get_mobility_specific_period

scenario_start_time = [505, 476, 556, 481, 640, 640]  # 505 - 20 May, 476 - 21st April,
# 556 - 9th July 2021, 481 - 25 April, 650 - Oct 01
lockdown_title = ["No lockdowns placed", "What if lockdown was initiated from April 21 - June 21",
                  "What if lockdown was initiated from July 10 - Oct 01", "No vaccination",
                  "Slower increase in mobility after lockdown ends on 01st October",
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
    num_scenarios = 4
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


            # from May 21 - 21 June, the average mobility observed before May 21
            times1 = [*range(507, 539)]
            values1 = {'work': [0.68] * len(times1), 'other_locations': [0.78] * len(times1)}

            # from August 21 - 01st October the average mobility observed before August 21
            times2 = [*range(599, 641)]
            values2 = {'work': [0.65] * len(times2), 'other_locations': [0.85] * len(times2)}

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 + times2 + [times2[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values2[key_loc] +
                              [["repeat_prev"]]
                }

        if i_lockdown_scenario == 1:  # What if lockdown was initiated from April 29 - May 29

            # lockdown mobility from 21May -21 June applied from April 29 - May 29
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [507, 537])

            times1 = [*range(485, 515)]  # lockdown values from May 21 - June 21 is assigned from April 29 -May 29

            # from May 30 - 21 June, the average mobility after lockdown
            times2 = [*range(515, 539)]
            values2 = {'work': [0.65] * len(times2), 'other_locations': [0.8] * len(times2)}

            # From June 22nd onwards actual mobility levels
            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown)
            # from Sep 01 st onwards

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
                                                            'home': {'residential': 1.}}, [599, 640])
            times1 = [*range(568, 609)]  # July 21 - 31 August

            times2 = [*range(609, 640)]
            values2 = {'work': [0.71] * len(times2), 'other_locations': [0.96] * len(times2)}

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown) frm Sep 01 st onwards
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

        if i_lockdown_scenario == 4:  # "Slower increase in mobility after lockdown ends on 01st October"
            for key_loc in ["other_locations", "work"]:
                if key_loc == "other_locations":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1] +
                                 [671, 672, 701, 702, 731, 732],
                        "values": [["repeat_prev"]] + [0.85] + [0.85, 0.9, 0.9, 0.95, 0.95, 1.04]
                    }
                if key_loc == "work":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1] +
                                 [671, 672, 701, 702, 731, 732],
                        "values": [["repeat_prev"]] + [0.7] + [0.7, 0.75, 0.75, 0.8, 0.8, 1.00]
                    }
            # scenario 3 is the same mobility as baseline but no vaccination

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