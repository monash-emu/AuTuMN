import os
from WPR_constants import WPR_Countries
from autumn.core.project import build_rel_path
import shutil
import re

for country in WPR_Countries:
    # If a country folder does not exist generate one
    directory_path = os.getcwd()
    folder_name = os.path.basename(directory_path)
    folder_path = build_rel_path(f"{country}/{country}")
    params_path = build_rel_path(f"{country}/{country}"+"/params")

    # Check whether the specified path exists or not
    isExist = os.path.exists(folder_path)
    isExistbaseline = os.path.exists(params_path)

    if not isExist:
        os.makedirs(folder_path)  # Create a new directory because it does not exist
    if not isExistbaseline:
        os.makedirs(params_path)

    # generate a project file
    file_name = "project.py"
    project_file_path = os.path.join(folder_path, file_name)
    project_file_path = f"{project_file_path}"

    # writing generate_project_WPR into country specific project file
    source_path = build_rel_path("generate_project_WPR.py")
    source_path = f"{source_path}"
    shutil.copy(source_path, project_file_path)

    # in each project file replacing the country name to corresponding country
    with open(project_file_path, 'r+') as project_file:
        read_file = project_file.read()
        project_file.seek(0)

        read_file = re.sub("country_name", f'"{country}"', read_file)
        project_file.write(read_file)
        project_file.truncate()

    # generate a timeseries file
    timeseries_file_name = "timeseries.json"
    complete_timeseries_name = os.path.join(folder_path, timeseries_file_name)
    timeseries_file = open(complete_timeseries_name, "w")
    # writing generate_project_WPR into country specific project file
    source_path_timeseries = build_rel_path("timeseries_dummy.txt")
    source_path_timeseries = f"{source_path_timeseries}"
    shutil.copy(source_path_timeseries, complete_timeseries_name)

    # create baseline file
    baseline_file_name = "baseline.yml"
    baseline_file_path = os.path.join(params_path, baseline_file_name)
    baseline_file_path = f"{baseline_file_path}"

    # writing basic baseline content into country specific baseline file
    source_path_baseline = build_rel_path("generate_baseline_file.py")
    source_path_baseline = f"{source_path_baseline}"
    shutil.copy(source_path_baseline, baseline_file_path)




