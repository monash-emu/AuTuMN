"""add comment in script explaining what its for
This is where the scripts to prepross the data go
save files in data/targets/
"""
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
import itertools
from google_drive_downloader import GoogleDriveDownloader as gdd
import json

# shareable google drive links
PHL_doh_link = "1gcFfBP-ZZRLfyK1E5g3DXW0047CmBOBa"  # sheet 05 daily report
PHL_fassster_link = "1DVhGsnHeiGEMJBPi0Wz5wJOrJAVMFXoV"

# destination folders filepaths
base_dir = os.path.dirname(os.path.abspath(os.curdir))
PHL_doh_dest = "./data/targets/PHL_icu.csv"
PHL_fassster_dest = "./data/targets/PHL_ConfirmedCases.zip"
icu_dest = "./data/targets/PHL_icu_processed.csv"
deaths_dest = "./data/targets/PHL_deaths_processed.csv"
notifications_dest = "./data/targets/PHL_notifications_processed.csv"
targets_dir = "./data/targets/"

# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

# function to fetch data
def fetch_phl_data():
    gdd.download_file_from_google_drive(file_id=PHL_doh_link, dest_path=PHL_doh_dest)
    gdd.download_file_from_google_drive(
        file_id=PHL_fassster_link, dest_path=PHL_fassster_dest, unzip=True
    )
    os.remove(PHL_fassster_dest)  # remove zip folder


# function to preprocess Philippines data for calibration targets
def process_phl_data():
    # read csvs
    up_to_data_fassster_filename = [
        filename
        for filename in os.listdir(targets_dir)
        if filename.startswith("ConfirmedCases_Final_")
    ]
    up_to_data_fassster_filename = targets_dir + str(up_to_data_fassster_filename[-1])
    fassster_data = pd.read_csv(up_to_data_fassster_filename)
    doh_data = pd.read_csv(PHL_doh_dest)
    # rename regions
    doh_data["region"] = doh_data["region"].replace(
        {
            "NATIONAL CAPITAL REGION (NCR)": "manila",
            "REGION IV-A (CALABAR ZON)": "calabarzon",
            "REGION VII (CENTRAL VISAYAS)": "central_visayas",
        }
    )
    fassster_data["Region"] = fassster_data["Region"].replace(
        {"NCR": "manila", "4A": "calabarzon", "07": "central_visayas"}
    )
    # duplicate data to create 'philippines' region and join with original dataset
    doh_data_dup = doh_data.copy()
    fassster_data_dup = fassster_data.copy()
    doh_data_dup["region"] = "philippines"
    fassster_data_dup["Region"] = "philippines"
    doh_data = doh_data.append(doh_data_dup)
    fassster_data = fassster_data.append(fassster_data_dup)
    # filter by regions (exclude all regions not modeled)
    regions = ["calabarzon", "central_visayas", "manila", "philippines"]
    doh_data = doh_data[doh_data["region"].isin(regions)]
    fassster_data = fassster_data[fassster_data["Region"].isin(regions)]
    ## most recent ICU data
    doh_data.loc[:, "reportdate"] = pd.to_datetime(doh_data["reportdate"])
    doh_data["times"] = doh_data.reportdate - COVID_BASE_DATETIME
    doh_data["times"] = doh_data["times"] / np.timedelta64(1, "D")
    icu_occ = doh_data.groupby(["region", "times"], as_index=False).sum(min_count=1)[
        ["region", "times", "icu_o"]
    ]
    icu_occ.to_csv(icu_dest)
    ## accumulated deaths
    fassster_data_deaths = fassster_data[fassster_data["Date_Died"].notna()]
    fassster_data_deaths.loc[:, "Date_Died"] = pd.to_datetime(fassster_data_deaths["Date_Died"])
    fassster_data_deaths.loc[:, "times"] = (
        fassster_data_deaths.loc[:, "Date_Died"] - COVID_BASE_DATETIME
    )
    fassster_data_deaths["times"] = fassster_data_deaths["times"] / np.timedelta64(
        1, "D"
    )  # warning
    accum_deaths = fassster_data_deaths.groupby(["Region", "times"]).size()
    accum_deaths = accum_deaths.to_frame(name="daily_deaths").reset_index()
    accum_deaths["accum_deaths"] = accum_deaths["daily_deaths"].cumsum()
    cumulative_deaths = accum_deaths[["Region", "times", "accum_deaths"]]
    cumulative_deaths.to_csv(deaths_dest)
    ## notifications
    fassster_data_agg = fassster_data.groupby(["Region", "imputed_Date_Admitted"]).size()
    fassster_data_agg = fassster_data_agg.to_frame(name="daily_notifications").reset_index()
    fassster_data_agg["imputed_Date_Admitted"] = pd.to_datetime(
        fassster_data_agg["imputed_Date_Admitted"]
    )
    # make sure all dates within range are included
    fassster_data_agg["times"] = fassster_data_agg.imputed_Date_Admitted - COVID_BASE_DATETIME
    fassster_data_agg["times"] = fassster_data_agg["times"] / np.timedelta64(1, "D")
    timeIndex = np.arange(
        min(fassster_data_agg["times"]), max(fassster_data_agg["times"]), 1.0
    ).tolist()
    all_regions_x_times = pd.DataFrame(
        list(itertools.product(regions, timeIndex)), columns=["Region", "times"]
    )
    fassster_agg_complete = pd.merge(
        fassster_data_agg, all_regions_x_times, on=["Region", "times"], how="outer"
    )
    fassster_agg_complete.loc[
        fassster_agg_complete["daily_notifications"].isna() == True, "daily_notifications"
    ] = 0
    # calculate a 7-day rolling window value
    fassster_agg_complete = fassster_agg_complete.sort_values(
        ["Region", "times"], ascending=[True, True]
    )
    fassster_agg_complete["mean_daily_notifications"] = (
        fassster_agg_complete.groupby("Region")
        .rolling(7)["daily_notifications"]
        .mean()
        .reset_index(0, drop=True)
    )
    fassster_agg_complete["mean_daily_notifications"] = np.round(
        fassster_agg_complete["mean_daily_notifications"]
    )
    fassster_data_final = fassster_agg_complete[fassster_agg_complete.times > 60]
    fassster_data_final = fassster_data_final[
        fassster_data_final.times < max(fassster_data_final.times) - 9
    ]
    fassster_data_final.to_csv(notifications_dest)
    # remove pre-processed files
    os.remove(up_to_data_fassster_filename)
    os.remove(PHL_doh_dest)


phl_regions = ["calabarzon", "central_visayas", "manila", "philippines"]


def update_calibration_phl():
    # read in csvs
    icu = pd.read_csv(icu_dest)
    deaths = pd.read_csv(deaths_dest)
    notifications = pd.read_csv(notifications_dest)
    for region in phl_regions:
        icu_tmp = icu.loc[icu["region"] == region]
        deaths_tmp = deaths.loc[deaths["Region"] == region]
        notifications_tmp = notifications.loc[notifications["Region"] == region]
        file_path = os.path.join(
            base_dir + "\\apps\\covid_19\\regions\\" + region + "\\targets.json"
        )

        with open(file_path, mode="r") as f:
            targets = json.load(f)

            targets["notifications"]["times"] = list(notifications_tmp["times"])
            targets["notifications"]["values"] = list(notifications_tmp["mean_daily_notifications"])
            targets["icu_occupancy"]["times"] = list(icu_tmp["times"])
            targets["icu_occupancy"]["values"] = list(icu_tmp["icu_o"])
            targets["infection_deaths"]["times"] = list(deaths_tmp["times"])
            targets["infection_deaths"]["values"] = list(deaths_tmp["accum_deaths"])

        with open(file_path, "w") as f:
            json.dump(targets, f, indent=2)
