import yaml
import os
from copy import copy
from autumn.region import Region


def read_all_phl_scenarios():
    """
    Read all the scenarios defined for the "philippines" application
    :return: a dictionary containing all the scenario parameters
    """
    scenario_param_dicts = {}

    param_files = os.listdir("params/")
    for filename in param_files:
        if filename.startswith("scenario-"):
            file_path = f"params/{filename}"
            with open(file_path) as file:
                sc_dict = yaml.load(file)

            scenario_param_dicts[filename] = sc_dict

    return scenario_param_dicts


def copy_scenarios_to_phl_regions():
    """
    Replicate all scenarios defined for the "philippines" application to the three regional applications
    :return:
    """
    scenario_param_dicts = read_all_phl_scenarios()

    for region in Region.PHILIPPINES_REGIONS:
        if region == "philippines":
            continue
        dir_name = region.replace("-", "_")

        for filename, sc_dict in scenario_param_dicts.items():
            region_scenario_param = copy(sc_dict)
            region_scenario_param['parent'] = sc_dict['parent'].replace("philippines", region)

            file_path = f"../{dir_name}/params/{filename}"
            with open(file_path, "w") as f:
                yaml.dump(region_scenario_param, f)


if __name__ == "__main__":
    copy_scenarios_to_phl_regions()
