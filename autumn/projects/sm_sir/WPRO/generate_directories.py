import os
#from WPR_constants import WPR_Countries
from autumn.core.project import build_rel_path
import shutil

WPR_Countries = [
    "australia",
    "brunei",
    "philippines",
    "malaysia"
]

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
    project_file_path = os.path.join(folder_path, file_name)
    project_file_path = f"{project_file_path}"

    # generate a timeseries file
    timeseries_file_name = "timeseries.json"
    complete_timeseries_name = os.path.join(folder_path, timeseries_file_name)
    timeseries_file = open(complete_timeseries_name, "w")

    # writing generate_project_WPR into country specific project file
    source_path = build_rel_path("generate_project_WPR.py")
    source_path = f"{source_path}"
    shutil.copy(source_path, project_file_path)

    country_name = country

