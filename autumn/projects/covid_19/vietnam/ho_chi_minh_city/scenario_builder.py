# Lockdown 1: Lockdown is maintained until 15/9, then relaxes slowly every 2 weeks (google mobility increases 20% each 2 weeks).
# Lockdown 2: Lockdown is maintained until 31/8, then relaxes slowly every 2 weeks (same as scenario 1).

scenario_start_time = 599  # 21 Aug 2021
end_lockdowns = [624, 609]
i_vacc_scenarios = range(4)

date_end_lockdowns = {609: "31 Aug 2021"}


def get_mobility_values(end_lockdown):
    times = [scenario_start_time, end_lockdown]
    values = [["repeat_prev"], ["repeat_prev"]]

    time = end_lockdown
    for _ in range(5): # probably fewer iterations required but it doesn't matter
        times += [
            time + 2,
            time + 14
        ]
        values += [
            ["add_to_prev_up_to_1", 0.20],
            ["repeat_prev"]
        ]

        time += 14

    return times, values


# Vaccine scenarios
periods = [
    [597, 624], # 19/8 - 15/9
    [625, 654], # 16/9 - 15/10
    [655, 731], # 16/10 - 31/12
    [732, 821], # 1/1 - 31/3
    [822, 912] # 1/4 - 30/6
]

roll_outs = [
    {
        "age_0": [0, 0, 0, 0, 0.8],
        "age_20": [0.061763957, 0.216851336, 0.138448386, 0.459133045, 0.352405081],
        "age_65": [0.468349415, 0.772826338, 0, 0, 0]
    },
    {
        "age_0": [0, 0, 0, 0, 0.8],
        "age_20": [0.098095696, 0.225586838, 0.145650183, 0.365315857, 0.422097035],
        "age_65": [0.117087354, 0.132614879, 0.305780848 ,0.440467318, 0.593993935]
    },
    {
        "age_0": [0, 0, 0, 0, 0.8],
        "age_20": [0.054497609, 0.215184797, 0.137092653, 0.431226633, 0.438940251],
        "age_65": [0.819611476, 0.324542136, 0, 0, 0]
    },
    {
        "age_0": [0, 0, 0, 0, 0.8],
        "age_20": [0.098095696, 0.112793419, 0.090809434, 0.2796624, 0.138656414],
        "age_65": [0.117087354, 0.132614879, 0.152890424, 0.360969651, 0.282435452]
    },
]


def get_vaccine_roll_out(i_vacc_scenario):
    age_mins = [0, 20, 65]
    roll_out_components = []
    for i_age_min, age_min in enumerate(age_mins):
        for i_period, period in enumerate(periods):
            coverage = roll_outs[i_vacc_scenario][f"age_{age_min}"][i_period]
            if coverage > 0.:
                component = {
                    "supply_period_coverage":
                        {"coverage": coverage, "start_time": period[0], "end_time": period[1]},
                }
                if age_min > 0:
                    component["age_min"] = age_min
                if age_min < 65:
                    component["age_max"] = age_mins[i_age_min + 1] - 1

                roll_out_components.append(component)

    return roll_out_components


def get_all_scenario_dicts():

    all_scenario_dicts = []
    for i_lockdown_scenario in [0, 1]:
        lockdown_title = f"Lockdown relaxed on {list(date_end_lockdowns.values())[0]}"


        if i_lockdown_scenario == 1:
            lockdown_title += " 60% reduced mobility after that."
        else:
            lockdown_title += " back to normal mobility after that."

        for i_vacc_scenario in i_vacc_scenarios:

            vaccine_title = f"Vaccine scenario V{i_vacc_scenario + 1}"
            scenario_dict = {
                "time": {"start": scenario_start_time},
                "description": f"{lockdown_title} / {vaccine_title}",
                "mobility": {"mixing": {}},
                "vaccination": {"roll_out_components": []}
            }

            # mobility parameters
            if i_lockdown_scenario == 0:
                times, values = get_mobility_values(609)
                for key_loc in ["other_locations", "work"]:
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": times,
                        "values": values
                    }
            else:
                for key_loc in ["other_locations", "work"]:
                    scenario_dict["mobility"]["mixing"][key_loc] = {
                        "append": True,
                        "times": [scenario_start_time, 609, 609 + 2, 609 + 14, 609 + 14 + 2],
                        "values": [["repeat_prev"], ["repeat_prev"], .40, .40, .60]
                    }
            # scenario_dict["mobility"]["mixing"]["school"] = {
            #         "append": False,
            #         "times": [end_lockdown, end_lockdown + 2],
            #         "values": [0, 1]
            # }

            # vaccination parameters
            scenario_dict["vaccination"]["roll_out_components"] = get_vaccine_roll_out(i_vacc_scenario)

            all_scenario_dicts.append(scenario_dict)

    return all_scenario_dicts
