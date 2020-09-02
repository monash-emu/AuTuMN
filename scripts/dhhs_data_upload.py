import pandas as pd
import os
import json
from datetime import datetime
from autumn import constants
from autumn import secrets


DHHS_DATA = os.path.join(constants.INPUT_DATA_PATH, "monashmodelextract_20200901.csv")
REGION_FOLDER = os.path.join(constants.APPS_PATH,"covid_19\\regions")
IMPORT_FOLDER = os.path.join(constants.INPUT_DATA_PATH, "imports")

CLUSTER_MAP = {1:"NORTH_METRO", 2:"SOUTH_EAST_METRO", 3:"SOUTH_METRO", 4:"WEST_METRO", 5:"BARWON_SOUTH_WEST",
               6:"GIPPSLAND", 7:"GRAMPIANS", 8:"HUME", 9:"LODDON_MALLEE"}


dhhs_df = pd.read_csv(DHHS_DATA)
dhhs_df.date = pd.to_datetime(dhhs_df["date"], infer_datetime_format=True)


def prep_df(df=dhhs_df,acq=int):

    df = df[dhhs_df.acquired == acq][["date","cluster", "new", "deaths",
                                                "incident_ward", "ward", "incident_icu", "icu"]]

    df = df.groupby(["date","cluster"]).sum().reset_index()
    df["cluster_name"] = df.cluster
    df["cluster_name"] = df.cluster_name.replace(CLUSTER_MAP).str.lower()
    df["date_index"] = (df.date - pd.datetime(2020,1,1)).dt.days
    df = df[(df.date_index != 244) &(df.date_index != 243)]

    return df


def update_calibration(cal_df):

    cal_df = prep_df(cal_df, acq=1)

    for region in CLUSTER_MAP:

        current_cluster = CLUSTER_MAP[region].lower()
        update_df = cal_df[cal_df.cluster_name == current_cluster]

        FILE_PATH = REGION_FOLDER + "\\" + current_cluster + "\\targets.secret.json"

        with open(FILE_PATH, mode='r') as TARGET_JSON:
            update_frame = json.load(TARGET_JSON)
    

        update_frame['notifications']['times'] = list(update_df.date_index)
        update_frame['notifications']['values'] = list(update_df.new)
        update_frame['hospital_occupancy']['times'] = list(update_df.date_index)
        update_frame['hospital_occupancy']['values'] = list(update_df.ward)
        update_frame['icu_occupancy']['times'] = list(update_df.date_index)
        update_frame['icu_occupancy']['values'] = list(update_df.icu)
        update_frame['total_infection_deaths']['times'] = list(update_df.date_index)
        update_frame['total_infection_deaths']['values'] = list(update_df.deaths)
        update_frame['icu_admissions']['times'] = list(update_df.date_index)
        update_frame['icu_admissions']['values'] = list(update_df.incident_icu)
        update_frame['hospital_admissions']['times'] = list(update_df.date_index)
        update_frame['hospital_admissions']['values'] = list(update_df.incident_ward)

        with open(FILE_PATH, mode='w')as TARGET_JSON:
            json.dump(update_frame,TARGET_JSON)

        secrets.write(FILE_PATH,'superspreader')





def update_importation(imp_df):

    imp_df = prep_df(imp_df, acq=4)

    for region in CLUSTER_MAP:
        current_cluster = CLUSTER_MAP[region].lower()
        update_df = imp_df[imp_df.cluster_name == current_cluster]

        FILE_PATH = IMPORT_FOLDER + "\\" + current_cluster + ".secret.json"
                
        with open(FILE_PATH, mode='r') as TARGET_JSON:
            update_frame = json.load(TARGET_JSON)
    
        update_frame['notifications']['times'] = list(update_df.date_index)
        update_frame['notifications']['values'] = list(update_df.new)


        with open(FILE_PATH, mode='w')as TARGET_JSON:
            json.dump(update_frame,TARGET_JSON)

        secrets.write(FILE_PATH,'superspreader')
        

if __name__ == "__main__":
         update_calibration(dhhs_df)
         update_importation(dhhs_df)