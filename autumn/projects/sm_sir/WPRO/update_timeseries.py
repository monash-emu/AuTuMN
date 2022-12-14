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
