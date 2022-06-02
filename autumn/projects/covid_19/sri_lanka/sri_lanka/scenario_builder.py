from autumn.models.sm_sir.mixing_matrix.macrodistancing import get_mobility_specific_period

scenario_start_time = [505, 476, 556, 481, 640, 640]  # 505 - 20 May, 476 - 21st April,
# 556 - 9th July 2021, 481 - 25 April, 650 - Oct 01
lockdown_title = ["No lockdowns placed", "What if lockdown was initiated from April 21 - June 21",
                  "What if lockdown was initiated from July 10 - Oct 01", "No vaccination",
                  "Slower increase in mobility after lockdown ends on 01st October",
                  "Faster increase in mobility after lockdown ends on 01st October"]


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
    num_scenarios = 1
    all_scenario_dicts = []

    for i_lockdown_scenario in [*range(0, num_scenarios)]:

        scenario_dict = {
            "time": {"start": scenario_start_time[i_lockdown_scenario]},
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
                    "values": [["repeat_prev"]] + [1.1]
                }

        if i_lockdown_scenario == 1:  # What if lockdown was initiated from April 21 - June 21

            # lockdown mobility from 21May -21 June
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [507, 537])

            times1 = [*range(477, 507)]  # lockdown values from May 21 - June 21 is assigned from April 21 -May 20

            # from May 20 - 21 June, the average mobility from from May 21 - June 21
            times5 = [*range(507, 539)]
            values5 = {'work': [0.4] * len(times5), 'other_locations': [0.4] * len(times5)}

            # mobility from 22 June -21st August
            times2, values2 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [539, 599])

            # In the scenario from August 21 - Oct 01 applying average mobility observed after lockdown
            # workplaces: 0.65 and other locations 0.8
            times3 = [*range(599, 640)]
            values3 = {'work': [0.65] * len(times3), 'other_locations': [0.8] * len(times3)}

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown)
            times4, values4 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [640, 645])

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

            # In the scenario, from July 10 - Aug 21 applying average mobility observed during lockdown
            # workplaces: 0.4 and other locations 0.4
            times3 = [*range(557, 599)]
            values3 = {'work': [0.4] * len(times3), 'other_locations': [0.4] * len(times3)}

            # lockdown mobility from 21Aug -01 Oct is assigned from  Aug 21 - Oct 01 in the scenario
            times1, values1 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [599, 641])

            # In the scenarios applying the actual values observed from Oct 02 -Oct 12 (after lockdown)
            times4, values4 = get_mobility_specific_period(country, None,
                                                           {'work': {'workplaces': 1.},
                                                            'other_locations': {'retail_and_recreation': 0.333,
                                                                                'grocery_and_pharmacy': 0.333,
                                                                                'transit_stations': 0.334},
                                                            'home': {'residential': 1.}}, [641, 645])

            for key_loc in ["other_locations", "work"]:
                scenario_dict["mobility"]["mixing"][key_loc] = {
                    "append": True,
                    "times": [scenario_start_time[i_lockdown_scenario]] + times1 +
                             times3 + times4 + [times4[-1] + 1],
                    "values": [["repeat_prev"]] + values1[key_loc] + values3[key_loc] +
                              values4[key_loc] + [["repeat_prev"]]
                }
        if i_lockdown_scenario == 4:  # "Slower increase in mobility after lockdown ends on 01st October"
            for key_loc in ["other_locations", "work"]:
                if key_loc == "other_locations":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1] +
                                 [671, 672, 701, 702, 762, 763],
                        "values": [["repeat_prev"]] + [0.75] + [0.75, 0.8, 0.8, 0.85, 0.85, 1.04]
                    }
                if key_loc == "work":
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time[i_lockdown_scenario]] + [
                            scenario_start_time[i_lockdown_scenario] + 1] +
                                 [671, 672, 701, 702, 762, 763],
                        "values": [["repeat_prev"]] + [0.6] + [0.6, 0.65, 0.65, 0.7, 0.7, 1.00]
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

        all_scenario_dicts.append(scenario_dict)
    return all_scenario_dicts

