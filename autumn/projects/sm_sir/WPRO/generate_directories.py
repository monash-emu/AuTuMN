import os
from autumn.core.project import build_rel_path
from autumn.core.inputs.demography.queries import get_iso3_from_country_name
import shutil
import re
import yaml
import json
from autumn.projects.sm_sir.WPRO.generate_timeseries import get_timeseries_data
from pathlib import Path

wpro_list = json.load(open(Path(__file__).parent.parent.parent.parent / "wpro_list.json"))
WPR_Countries = wpro_list["region"]

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

    if not os.path.exists(project_file_path):
        # writing generate_project_WPR into country specific project file
        source_path = build_rel_path("generate_project_WPR.py")
        source_path = f"{source_path}"
        shutil.copy(source_path, project_file_path)

        # in each project file replacing the country name to corresponding country
        with open(project_file_path, 'r+') as project_file:
            read_file = project_file.read()
            project_file.seek(0)

            country_text = country
            if '-' in country_text:
                country_text = country_text.replace('-', '_')
            region_name = country_text.upper() # get the corresponding iso3 code

            read_file = re.sub("country_name", f'"{country}"', read_file)
            read_file = re.sub("COUNTRY_NAME", f"{region_name}", read_file)
            project_file.write(read_file)
            project_file.truncate()

    country_text = country

    if country == "south-korea":
        iso3_name = "KOR"
    elif country == "vietnam":
        iso3_name = "VNM"
    else:
        if '-' in country_text:
            country_text = country_text.replace('-', " ")
        iso3_name = get_iso3_from_country_name(f"{country_text.title()}")  # get the corresponding iso3 code

    # generate timeseries.json file
    timeseries_file_name = "timeseries.json"
    complete_timeseries_name = os.path.join(folder_path, timeseries_file_name)
    with open(complete_timeseries_name, "w") as timefile:
        timeseries_data = get_timeseries_data(iso3_name)
        json.dump(timeseries_data, timefile)

    # create baseline.yml file
    baseline_file_name = "baseline.yml"
    baseline_file_path = os.path.join(params_path, baseline_file_name)
    baseline_file_path = f"{baseline_file_path}"

    # in the yml file update iso3 to corresponding country value
    with open("generate_baseline_file.yml", 'r') as stream:
        load_yml_params = yaml.safe_load(stream)

    # Modify the fields from the dict
    load_yml_params['country']['iso3'] = iso3_name
    # Save the yml file again
    with open("generate_baseline_file.yml", 'w') as stream:
        yaml.dump(load_yml_params, stream, default_flow_style=False, sort_keys=False)

    # writing basic baseline content into country specific baseline file
    source_path_baseline = build_rel_path("generate_baseline_file.yml")
    source_path_baseline = f"{source_path_baseline}"

    # check if the baseline file exists and if so if it is empty
    if os.path.exists(baseline_file_path) and os.stat(baseline_file_path).st_size == 0:
        shutil.copyfile(source_path_baseline, baseline_file_path)
    if not os.path.exists(baseline_file_path):  # if baseline file is created for the first time
        shutil.copyfile(source_path_baseline, baseline_file_path)
