import pandas as pd
import os
import json
from datetime import datetime
from autumn import constants
from autumn import secrets


DHHS_DATA = os.path.join(constants.INPUT_DATA_PATH, "monashmodelextract_20200901.csv")
REGION_FOLDER  = os.path.join(constants.APPS_PATH,"covid_19\\regions")

CLUSTER_MAP = {1:"NORTH_METRO", 2:"SOUTH_EAST_METRO", 3:"SOUTH_METRO", 4:"WEST_METRO", 5:"BARWON_SOUTH_WEST",
               6:"GIPPSLAND", 7:"GRAMPIANS", 8:"HUME", 9:"LODDON_MALLEE"}


dhhs_df = pd.read_csv(DHHS_DATA)

dhhs_df.date = pd.to_datetime(dhhs_df["date"], infer_datetime_format=True)
dhhs_df = dhhs_df[dhhs_df.acquired==1][["date","cluster", "new", "deaths",
       "incident_ward", "ward", "incident_icu", "icu"]]
dhhs_df = dhhs_df.groupby(["date","cluster"]).sum().reset_index()
dhhs_df["cluster_name"] = dhhs_df.cluster
dhhs_df["cluster_name"] = dhhs_df.cluster_name.replace(CLUSTER_MAP).str.lower()
dhhs_df["date_index"] = (dhhs_df.date - pd.datetime(2020,1,1)).dt.days



for region in CLUSTER_MAP:
    current_cluster = CLUSTER_MAP[region].lower()
    update_df = dhhs_df[dhhs_df.cluster_name== current_cluster]

    with open(REGION_FOLDER+"\\"+current_cluster+"\\targets.secret.json", mode='r') as TARGET_JSON:
        update_frame = json.load(TARGET_JSON)
    

    update_frame['notifications']['times'] = list(update_df.date_index)
    update_frame['notifications']['values'] = list(update_df.new)
    update_frame['hospital_occupancy']['times'] = list(update_df.date_index)
    update_frame['hospital_occupancy']['values'] = list(update_df.incident_ward)
    update_frame['icu_occupancy']['times'] = list(update_df.date_index)
    update_frame['icu_occupancy']['values'] = list(update_df.incident_icu)
    update_frame['total_infection_deaths']['times'] = list(update_df.date_index)
    update_frame['total_infection_deaths']['values'] = list(update_df.deaths)

    with open(REGION_FOLDER+"\\"+current_cluster+"\\targets.secret.json", mode='w')as TARGET_JSON:
        json.dump(update_frame,TARGET_JSON)

