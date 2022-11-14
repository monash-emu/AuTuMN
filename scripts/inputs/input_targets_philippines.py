"""add comment in script explaining what its for
This is where the scripts to preprocess the data go
save files in data/targets/
"""
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.settings import INPUT_DATA_PATH, PROJECTS_PATH

PROJECTS_PATH = Path(PROJECTS_PATH)
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

# shareable google drive links
PHL_doh_link = "1j3X7-9Iry0c7RIH9GgO7RhhBt5Gn7LYR"  # sheet 05 daily report
PHL_fassster_link = "1PhZcRIngr6Xq2hlJJvmHpoiqsIOmxjFl"

# destination folders filepaths
phl_inputs_dir = INPUT_DATA_PATH / "covid_phl"
PHL_doh_dest = phl_inputs_dir / "PHL_icu.csv"
PHL_fassster_dest = phl_inputs_dir / "PHL_ConfirmedCases.zip"
icu_o_dest = phl_inputs_dir / "PHL_icu_processed.csv"
hosp_o_dest = phl_inputs_dir / "PHL_hosp_processed.csv"
cml_deaths_dest = phl_inputs_dir / "PHL_deaths_processed.csv"
daily_deaths_dest = phl_inputs_dir / "daily_deaths.csv"
notifications_dest = phl_inputs_dir / "PHL_notifications_processed.csv"


def main():
    fetch_phl_data()
    fassster_filename = fassster_data_filepath()

    # Process DoH data
    working_df = pd.read_csv(PHL_doh_dest)  # copy_davao_city_to_region(PHL_doh_dest)
    working_df = rename_regions(
        working_df,
        "region",
        "NATIONAL CAPITAL REGION (NCR)",
        "REGION IV-A (CALABAR ZON)",
        "REGION VII (CENTRAL VISAYAS)",
        "REGION XI (DAVAO REGION)",
        "BARMM",
        "REGION VI (WESTERN VISAYAS)",
    )
    working_df = duplicate_data(working_df, "region")
    working_df = filter_df_by_regions(working_df, "region")
    process_occupancy_data(working_df, "icu")
    process_occupancy_data(working_df, "hospital")

    # Now fassster data
    working_df = pd.read_csv(fassster_filename)  # copy_davao_city_to_region(fassster_filename)
    working_df = rename_regions(
        working_df,
        "Region",
        "NCR",
        "4A",
        "07",
        "11",
        "BARMM",
        "06",
    )
    working_df = duplicate_data(working_df, "Region")
    working_df = filter_df_by_regions(working_df, "Region")
    process_accumulated_death_data(working_df)
    process_notifications_data(working_df)
    update_calibration_phl()
    remove_files(fassster_filename)


# function to fetch data
def fetch_phl_data():

    doh = f"https://drive.google.com/uc?id={PHL_doh_link}&confirm=t"
    faster = f"https://drive.google.com/uc?id={PHL_fassster_link}&confirm=t"

    pd.read_csv(doh).to_csv(PHL_doh_dest)

    req = requests.get(faster)
    with open(PHL_fassster_dest, "wb") as output_file:
        output_file.write(req.content)

    with ZipFile(PHL_fassster_dest) as z:
        filename = [each.filename for each in z.filelist if each.filename.startswith("2022")]
        if len(filename) == 1:
            with z.open(filename[0]) as f:
                pd.read_csv(f).to_csv(phl_inputs_dir / filename[0])


def fassster_data_filepath():
    fassster_filename = [
        filename
        for filename in phl_inputs_dir.glob("*")
        if filename.stem.startswith("ConfirmedCases_Final_") or filename.stem.startswith("2022")
    ]
    fassster_filename = fassster_filename[0]
    return fassster_filename


def rename_regions(
    df: pd.DataFrame, regionSpelling, ncrName, calName, cenVisName, davName, BarName, wesVisName
):
    # df = pd.read_csv(filePath)
    df[regionSpelling] = df[regionSpelling].replace(
        {
            ncrName: "national-capital-region",
            calName: "calabarzon",
            cenVisName: "central_visayas",
            davName: "davao_region",
            BarName: "barmm",
            wesVisName: "western-visayas",
        }
    )
    return df


def duplicate_data(df: pd.DataFrame, regionSpelling):
    # df = pd.read_csv(filePath)
    data_dup = df.copy()
    data_dup[regionSpelling] = "philippines"
    return df.append(data_dup)


def filter_df_by_regions(df: pd.DataFrame, regionSpelling):

    regions = [
        "calabarzon",
        "central_visayas",
        "national-capital-region",
        "davao_city",
        "davao_region",
        "barmm",
        "western-visayas",
        "philippines",
    ]
    return df[df[regionSpelling].isin(regions)]


def process_occupancy_data(df: pd.DataFrame, occupancy_type: str) -> None:

    if occupancy_type.lower() == "hospital":
        dest = hosp_o_dest
        col = "nonicu_o"
    elif occupancy_type.lower() == "icu":
        dest = icu_o_dest
        col = "icu_o"

    df.loc[:, "reportdate"] = pd.to_datetime(df["reportdate"]).dt.tz_localize(None)
    df["times"] = df.reportdate - COVID_BASE_DATETIME
    df["times"] = df["times"] / np.timedelta64(1, "D")
    df_occ = df.groupby(["region", "times"], as_index=False).sum(min_count=1)[
        ["region", "times", col]
    ]
    df_occ.to_csv(dest)

    return None


def process_accumulated_death_data(df: pd.DataFrame):
    fassster_data_deaths = df[df["Date_Died"].notna()]
    fassster_data_deaths.loc[:, "Date_Died"] = pd.to_datetime(fassster_data_deaths["Date_Died"])
    fassster_data_deaths.loc[:, "times"] = (
        fassster_data_deaths.loc[:, "Date_Died"] - COVID_BASE_DATETIME
    )
    fassster_data_deaths["times"] = fassster_data_deaths["times"] / np.timedelta64(
        1, "D"
    )  # warning
    deaths_df = fassster_data_deaths.groupby(["Region", "times"]).size()
    deaths_df = deaths_df.to_frame(name="daily_deaths").reset_index()
    deaths_df.to_csv(daily_deaths_dest, index=False)
    deaths_df["accum_deaths"] = deaths_df.groupby("Region")["daily_deaths"].transform(
        pd.Series.cumsum
    )
    cml_deaths_df = deaths_df[["Region", "times", "accum_deaths"]]
    cml_deaths_df.to_csv(cml_deaths_dest)


def process_notifications_data(df: pd.DataFrame):
    fassster_data_agg = df.groupby(["Region", "Report_Date"]).size()
    fassster_data_agg = fassster_data_agg.to_frame(name="daily_notifications").reset_index()
    fassster_data_agg["Report_Date"] = pd.to_datetime(fassster_data_agg["Report_Date"])
    # make sure all dates within range are included
    fassster_data_agg["times"] = fassster_data_agg.Report_Date - COVID_BASE_DATETIME
    fassster_data_agg["times"] = fassster_data_agg["times"] / np.timedelta64(1, "D")
    timeIndex = np.arange(
        min(fassster_data_agg["times"]), max(fassster_data_agg["times"]), 1.0
    ).tolist()
    regions = [
        "calabarzon",
        "central_visayas",
        "national-capital-region",
        "davao_city",
        "davao_region",
        "barmm",
        "western-visayas",
        "philippines",
    ]
    all_regions_x_times = pd.DataFrame(
        list(itertools.product(regions, timeIndex)), columns=["Region", "times"]
    )
    fassster_agg_complete = pd.merge(
        fassster_data_agg, all_regions_x_times, on=["Region", "times"], how="outer"
    )
    fassster_agg_complete.loc[
        fassster_agg_complete["daily_notifications"].isna() == True,
        "daily_notifications",
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


def write_to_file(icu_tmp, deaths_tmp, notifications_tmp, hosp_tmp, daily_deaths_tmp, file_path):
    with open(file_path, mode="r") as f:
        targets = json.load(f)

        targets["notifications"]["times"] = list(notifications_tmp["times"])
        targets["notifications"]["values"] = list(notifications_tmp["mean_daily_notifications"])
        # targets["icu_occupancy"]["times"] = list(icu_tmp["times"])
        # targets["icu_occupancy"]["values"] = list(icu_tmp["icu_o"])
        targets["cumulative_deaths"]["times"] = list(deaths_tmp["times"])
        targets["cumulative_deaths"]["values"] = list(deaths_tmp["accum_deaths"])
        targets["infection_deaths"]["times"] = list(daily_deaths_tmp["times"])
        targets["infection_deaths"]["values"] = list(daily_deaths_tmp["daily_deaths"])
        # targets["hospital_occupancy"]["times"] = list(hosp_tmp["times"])
        # targets["hospital_occupancy"]["values"] = list(hosp_tmp["nonicu_o"])

    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)


def update_calibration_phl():

    # Only ncr, barmm and western-visayas used by sm_sir model
    phl_regions = [
        # "calabarzon",
        # "central_visayas",
        "national-capital-region",
        "barmm",
        "western-visayas",
        # "davao_city",
        # "davao_region",
        # "philippines",
    ]
    # read in csvs
    icu_occ = pd.read_csv(icu_o_dest)
    deaths = pd.read_csv(cml_deaths_dest)
    notifications = pd.read_csv(notifications_dest)
    hosp_occ = pd.read_csv(hosp_o_dest)
    daily_deaths = pd.read_csv(daily_deaths_dest)

    for region in phl_regions:
        icu_tmp = icu_occ.loc[icu_occ["region"] == region]
        deaths_tmp = deaths.loc[deaths["Region"] == region]
        notifications_tmp = notifications.loc[notifications["Region"] == region]
        hosp_tmp = hosp_occ.loc[hosp_occ["region"] == region]
        daily_deaths_tmp = daily_deaths.loc[daily_deaths["Region"] == region]

        SM_SIR_TS = Path(
            PROJECTS_PATH,
            "sm_sir",
            "philippines",
            region,
            "timeseries.json",
        )

        write_to_file(
            icu_tmp,
            deaths_tmp,
            notifications_tmp,
            hosp_tmp,
            daily_deaths_tmp,
            SM_SIR_TS,
        )


def remove_files(filePath1):

    files = {
        filePath1,
        PHL_fassster_dest,
        PHL_doh_dest,
        icu_o_dest,
        cml_deaths_dest,
        notifications_dest,
        daily_deaths_dest,
        hosp_o_dest,
    }

    for file in files:
        file.unlink()


def copy_davao_city_to_region(filePath) -> pd.DataFrame:
    df = pd.read_csv(filePath)
    if filePath is PHL_doh_dest:
        df.loc[df.city_mun == "DAVAO CITY", ["region"]] = "davao_city"
    elif "ConfirmedCases_Final_" in filePath:
        df.loc[df.CityMunicipality == "DAVAO CITY", ["Region"]] = "davao_city"
    else:
        return 0
    return df


if __name__ == "__main__":
    main()
