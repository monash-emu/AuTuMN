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
from settings import APPS_PATH

# shareable google drive links
PHL_doh_link = "1x2A-rgFhfr706yCpjPNy8sHTVUPqalX8"  # sheet 05 daily report
PHL_fassster_link = "1wnE_p3NEdgaG5Oi_7x-s9mzkFeNC4VSh"

# destination folders filepaths
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


def fassster_data_filepath():
    fassster_filename = [
        filename
        for filename in os.listdir(targets_dir)
        if filename.startswith("ConfirmedCases_Final_")
    ]
    fassster_filename = targets_dir + str(fassster_filename[-1])
    return fassster_filename


def rename_regions(filePath, regionSpelling, ncrName, calName, cenVisName):
    df = pd.read_csv(filePath)
    df[regionSpelling] = df[regionSpelling].replace(
        {ncrName: "manila", calName: "calabarzon", cenVisName: "central_visayas"}
    )
    df.to_csv(filePath)


def duplicate_data(filePath, regionSpelling):
    df = pd.read_csv(filePath)
    data_dup = df.copy()
    data_dup[regionSpelling] = "philippines"
    newdf = df.append(data_dup)
    newdf.to_csv(filePath)


def filter_df_by_regions(filePath, regionSpelling):
    df = pd.read_csv(filePath)
    regions = ["calabarzon", "central_visayas", "manila", "philippines"]
    df_regional = df[df[regionSpelling].isin(regions)]
    df_regional.to_csv(filePath)


def process_icu_data():
    df = pd.read_csv(PHL_doh_dest)
    df.loc[:, "reportdate"] = pd.to_datetime(df["reportdate"])
    df["times"] = df.reportdate - COVID_BASE_DATETIME
    df["times"] = df["times"] / np.timedelta64(1, "D")
    icu_occ = df.groupby(["region", "times"], as_index=False).sum(min_count=1)[
        ["region", "times", "icu_o"]
    ]
    icu_occ.to_csv(icu_dest)


def process_accumulated_death_data(filePath):
    df = pd.read_csv(filePath)
    fassster_data_deaths = df[df["Date_Died"].notna()]
    fassster_data_deaths.loc[:, "Date_Died"] = pd.to_datetime(fassster_data_deaths["Date_Died"])
    fassster_data_deaths.loc[:, "times"] = (
        fassster_data_deaths.loc[:, "Date_Died"] - COVID_BASE_DATETIME
    )
    fassster_data_deaths["times"] = fassster_data_deaths["times"] / np.timedelta64(
        1, "D"
    )  # warning
    accum_deaths = fassster_data_deaths.groupby(["Region", "times"]).size()
    accum_deaths = accum_deaths.to_frame(name="daily_deaths").reset_index()
    accum_deaths["accum_deaths"] = accum_deaths.groupby("Region")["daily_deaths"].transform(
        pd.Series.cumsum
    )
    cumulative_deaths = accum_deaths[["Region", "times", "accum_deaths"]]
    cumulative_deaths.to_csv(deaths_dest)


def process_notifications_data(filePath):
    df = pd.read_csv(filePath)
    fassster_data_agg = df.groupby(["Region", "Report_Date"]).size()
    fassster_data_agg = fassster_data_agg.to_frame(name="daily_notifications").reset_index()
    fassster_data_agg["Report_Date"] = pd.to_datetime(fassster_data_agg["Report_Date"])
    # make sure all dates within range are included
    fassster_data_agg["times"] = fassster_data_agg.Report_Date - COVID_BASE_DATETIME
    fassster_data_agg["times"] = fassster_data_agg["times"] / np.timedelta64(1, "D")
    timeIndex = np.arange(
        min(fassster_data_agg["times"]), max(fassster_data_agg["times"]), 1.0
    ).tolist()
    regions = ["calabarzon", "central_visayas", "manila", "philippines"]
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
        fassster_data_final.times < max(fassster_data_final.times)
    ]
    fassster_data_final.to_csv(notifications_dest)


def update_calibration_phl():
    phl_regions = ["calabarzon", "central_visayas", "manila", "philippines"]
    # read in csvs
    icu = pd.read_csv(icu_dest)
    deaths = pd.read_csv(deaths_dest)
    notifications = pd.read_csv(notifications_dest)
    for region in phl_regions:
        icu_tmp = icu.loc[icu["region"] == region]
        deaths_tmp = deaths.loc[deaths["Region"] == region]
        notifications_tmp = notifications.loc[notifications["Region"] == region]
        file_path = os.path.join(APPS_PATH, "covid_19", "regions", region, "targets.json")

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


def remove_files(filePath1):
    os.remove(filePath1)
    os.remove(PHL_fassster_dest)
    os.remove(PHL_doh_dest)
    os.remove(icu_dest)
    os.remove(deaths_dest)
    os.remove(notifications_dest)


fetch_phl_data()
fassster_filename = fassster_data_filepath()
rename_regions(
    PHL_doh_dest,
    "region",
    "NATIONAL CAPITAL REGION (NCR)",
    "REGION IV-A (CALABAR ZON)",
    "REGION VII (CENTRAL VISAYAS)",
)
rename_regions(fassster_filename, "Region", "NCR", "4A", "07")
duplicate_data(PHL_doh_dest, "region")
duplicate_data(fassster_filename, "Region")
filter_df_by_regions(PHL_doh_dest, "region")
filter_df_by_regions(fassster_filename, "Region")
process_icu_data()
process_accumulated_death_data(fassster_filename)
process_notifications_data(fassster_filename)
update_calibration_phl()
remove_files(fassster_filename)
