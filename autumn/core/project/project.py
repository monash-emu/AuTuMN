import os
import glob
import re

def get_all_available_scenario_paths(scenario_dir_path):
    """
    Automatically lists the paths of all the yml files starting with 'scenario-' available in a given directory.
    :param scenario_dir_path: path to the directory
    :return: a list of paths
    """
    glob_str = os.path.join(scenario_dir_path, "scenario-*.yml")
    scenario_file_list = glob.glob(glob_str)

    # Sort by integer rather than string (so that 'scenario-2' comes before 'scenario-10')
    file_list_sorted = sorted(
        scenario_file_list, key=lambda x: int(re.match(".*scenario-([0-9]*)", x).group(1))
    )

    return file_list_sorted