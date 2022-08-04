import os
from WPR_constants import WPR_Countries
from autumn.core.project import build_rel_path
for country in WPR_Countries:
    # If a country folder does not exist generate one
    directory_path = os.getcwd()
    folder_name = os.path.basename(directory_path)
    folder_path = build_rel_path(f"{country}/{country}")

    # Check whether the specified path exists or not
    isExist = os.path.exists(folder_path)

    if not isExist:
        os.makedirs(folder_path)  # Create a new directory because it does not exist

    # generate a project file
    file_name = "project.py"
    complete_name = os.path.join(folder_path, file_name)
    project_file = open(complete_name, "w")

    # generate a timeseries file
    timeseries_file_name = "timeseries.json"
    complete_timeseries_name = os.path.join(folder_path, timeseries_file_name)
    timeseries_file = open(complete_timeseries_name, "w")

    # scanning each line in generate_project_WPR and writing into country specific project file
    with open("generate_project_WPR.py", "r") as scan:
        project_file.write(scan.read())