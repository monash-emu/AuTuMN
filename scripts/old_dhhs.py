#!/usr/bin/env python
"""
Script for loading DHHS data into calibration targets and import inputs.
"""
import os
import sys
import json
from datetime import datetime, time
from getpass import getpass

import numpy as np
import pandas as pd

# Do some super sick path hacks to get script to work from command line.
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from autumn import constants
from autumn import secrets


DHHS_CSV = os.path.join(constants.INPUT_DATA_PATH, "monashmodelextract.secret.csv")
CHRIS_CSV = os.path.join(constants.INPUT_DATA_PATH, "monitoringreport.secret.csv")
REGION_DIR = os.path.join(constants.APPS_PATH, "covid_19", "regions")
IMPORT_DIR = os.path.join(constants.INPUT_DATA_PATH, "imports")

CLUSTER_MAP = {
    1: "NORTH_METRO",
    2: "SOUTH_EAST_METRO",
    3: "SOUTH_METRO",
    4: "WEST_METRO",
    5: "BARWON_SOUTH_WEST",
    6: "GIPPSLAND",
    7: "GRAMPIANS",
    8: "HUME",
    9: "LODDON_MALLEE",
}

TARGETS_MAP = {
    "notifications": "new",
    "hospital_occupancy": "ward",
    "icu_occupancy": "icu",
    "infection_deaths": "deaths",
    "icu_admissions": "incident_icu",
    "hospital_admissions": "incident_ward",
}

ACQUIRED_LOCALLY = 1
ACQUIRED_OVERSEAS = 4
CHRIS_HOSPITAL = "Confirmed COVID ‘+’ cases admitted to your hospital"
CHRIS_ICU = "Confirmed COVID ‘+’ cases in your ICU/HDU(s)"

CHRIS_MAP = {
    "Royal Childrens Hospital [Parkville]": "WEST_METRO",
    "Alfred- The [Prahran]": "SOUTH_METRO",
    "Cabrini Malvern": "SOUTH_METRO",
    "Ballarat Health Services [Base Campus]": "GRAMPIANS",
    "Albury Wodonga Health - Albury": "HUME",
    "Epworth Freemasons": "WEST_METRO",
    "Sunshine Hospital": "WEST_METRO",
    "Western Hospital [Footscray]": "WEST_METRO",
    "St Vincents Hospital": "NORTH_METRO",
    "Bendigo Hospital- The": "LODDON_MALLEE",
    "Bays Hospital- The [Mornington]": "SOUTH_METRO",
    "Latrobe Regional Hospital [Traralgon]": "GIPPSLAND",
    "Peninsula Private Hospital [Frankston]": "SOUTH_METRO",
    "Royal Melbourne Hospital - City Campus": "WEST_METRO",
    "Melbourne Private Hospital- The [Parkville]": "WEST_METRO",
    "St John of God Geelong Hospital": "BARWON_SOUTH_WEST",
    "Maroondah Hospital [East Ringwood]": "SOUTH_EAST_METRO",
    "Frankston Hospital": "SOUTH_METRO",
    "St Vincents Private Hospital Fitzroy": "NORTH_METRO",
    "New Mildura Base Hospital": "LODDON_MALLEE",
    "Box Hill Hospital": "SOUTH_EAST_METRO",
    "Austin Hospital": "NORTH_METRO",
    "Angliss Hospital": "SOUTH_EAST_METRO",
    "Geelong Hospital": "BARWON_SOUTH_WEST",
    "Monash Medical Centre [Clayton]": "SOUTH_EAST_METRO",
    "Goulburn Valley Health [Shepparton]": "HUME",
    "Warringal Private Hospital [Heidelberg]": "NORTH_METRO",
    "St John of God Ballarat Hospital": "GRAMPIANS",
    "Epworth Eastern Hospital": "SOUTH_EAST_METRO",
    "South West Healthcare [Warrnambool]": "BARWON_SOUTH_WEST",
    "Northeast Health Wangaratta": "HUME",
    "Mercy Public Hospitals Inc [Werribee]": "WEST_METRO",
    "Epworth Hospital [Richmond]": "WEST_METRO",
    "Holmesglen Private Hospital ": "SOUTH_METRO",
    "Knox Private Hospital [Wantirna]": "SOUTH_EAST_METRO",
    "St John of God Bendigo Hospital": "LODDON_MALLEE",
    "Wimmera Base Hospital [Horsham]": "GRAMPIANS",
    "Valley Private Hospital- The [Mulgrave]": "SOUTH_EAST_METRO",
    "John Fawkner - Moreland Private Hospital": "WEST_METRO",
    "Epworth Geelong": "BARWON_SOUTH_WEST",
    "Monash Children's Hospital": "SOUTH_EAST_METRO",
    "Central Gippsland Health Service [Sale]": "GIPPSLAND",
    "Northern Hospital, The [Epping]": "NORTH_METRO",
    "Dandenong Campus": "SOUTH_EAST_METRO",
    "Hamilton Base Hospital": "BARWON_SOUTH_WEST",
    "St John of God Berwick Hospital": "SOUTH_EAST_METRO",
    "Casey Hospital": "SOUTH_EAST_METRO",
    "Mildura Base Public Hospital": "LODDON_MALLEE",
}


def main():
    password = os.environ.get(constants.PASSWORD_ENVAR, "")
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    update_calibration(password)
    update_importation(password)


def update_calibration(password: str):
    """
    Update values of Victorian cluster calibration targets
    """
    # Load locally acquired cases.
    cal_df = load_dhhs_df(ACQUIRED_LOCALLY)
    chris_df = load_chris_df(CHRIS_ICU)

    # Replace DHHS icu occupancy with CHRIS data
    cal_df = pd.merge(
        cal_df,
        chris_df,
        how="left",
        left_on=["cluster_name", "date_index"],
        right_on=["cluster_name", "date_index"],
    )
    cal_df.icu = cal_df.value

    vic_targets = {}
    for region in CLUSTER_MAP.keys():
        current_cluster = CLUSTER_MAP[region].lower()
        update_df = cal_df[cal_df.cluster_name == current_cluster]
        file_path = os.path.join(REGION_DIR, current_cluster, "targets.secret.json")
        with open(file_path, mode="r") as f:
            targets = json.load(f)

        for key, val in TARGETS_MAP.items():
            # Drop the NaN value rows from CHRIS data.
            temp_df = update_df[["date_index", val]].dropna(0, subset=[val])

            targets[key]["times"] = list(temp_df["date_index"])
            targets[key]["values"] = list(temp_df[val])

            # Add to VIC targets
            cluster = current_cluster.lower()
            vic_key = f"{key}_for_cluster_{cluster}"
            vic_targets[vic_key] = {
                "title": f"{targets[key]['title']} ({cluster})",
                "output_key": vic_key,
                "times": targets[key]["times"],
                "values": targets[key]["values"],
                "quantiles": targets[key]["quantiles"],
            }

        with open(file_path, "w") as f:
            json.dump(targets, f, indent=2)

        secrets.write(file_path, password)

    # Calculate VIC aggregate targets
    for key in TARGETS_MAP.keys():
        cluster_keys = [k for k in vic_targets.keys() if k.startswith(key)]

        times = set()
        for cluster_key in cluster_keys:
            cluster_times = vic_targets[cluster_key]["times"]
            times = times.union(cluster_times)

        min_time = min(times)
        max_time = max(times)
        times = [min_time + t for t in range(max_time - min_time + 1)]
        values = [0] * len(times)
        for cluster_key in cluster_keys:
            cluster_times = vic_targets[cluster_key]["times"]
            cluster_values = vic_targets[cluster_key]["values"]
            cluster_lookup = {t: v for t, v in zip(cluster_times, cluster_values)}
            for idx, t in enumerate(times):
                values[idx] += cluster_lookup.get(t, 0)

        # Re-use last targets from regional for loop
        vic_targets[key] = {
            "title": targets[key]["title"],
            "output_key": key,
            "times": times,
            "values": list(values),
            "quantiles": targets[key]["quantiles"],
        }

    # Write VIC aggregate secrets.
    file_path = os.path.join(REGION_DIR, "victoria", "targets.secret.json")
    with open(file_path, "w") as f:
        json.dump(vic_targets, f, indent=2)

    secrets.write(file_path, password)


def update_importation(password: str):
    """
    Update Victorian importation targets.
    """
    # Load imported cases
    imp_df = load_dhhs_df(ACQUIRED_OVERSEAS)
    imports_data = {}
    for region in CLUSTER_MAP.keys():
        current_cluster = CLUSTER_MAP[region].lower()
        update_df = imp_df[imp_df.cluster_name == current_cluster]
        region_name = get_region_name(current_cluster)
        imports_data[region_name] = {
            "description": f"Daily imports for {region_name}",
            "times": list(update_df.date_index),
            "values": list(update_df.new),
        }

    file_path = os.path.join(IMPORT_DIR, "imports.secret.json")
    with open(file_path, "w") as f:
        json.dump(imports_data, f, indent=2)

    secrets.write(file_path, password)


def get_region_name(cluster_name: str):
    return cluster_name.lower().replace("_", "-")


def load_dhhs_df(acquired: int):

    df = pd.read_csv(DHHS_CSV)
    df.date = pd.to_datetime(df["date"], infer_datetime_format=True)
    df = df[df.acquired == acquired][
        ["date", "cluster", "new", "deaths", "incident_ward", "ward", "incident_icu", "icu"]
    ]
    df = df.groupby(["date", "cluster"]).sum().reset_index()
    df["cluster_name"] = df.cluster
    df["cluster_name"] = df.cluster_name.replace(CLUSTER_MAP).str.lower()
    df["date_index"] = (df.date - datetime(2019, 12, 31)).dt.days

    # Remove last date due to poor accuracy of data.
    df = df[df.date_index != df.date_index.max()]

    return df


def load_chris_df(load: str):
    """
    Load data from CSV downloaded from CHRIS website
    """
    df = pd.read_csv(CHRIS_CSV)
    df.rename(
        columns={
            "Campus Name": "cluster_name",
            " Jurisdiction ": "state",
            "Field Name": "type",
            "Field Value": "value",
            "Effective From": "E_F",
            "Effective To": "E_T",
        },
        inplace=True,
    )

    df = df[df.type == load][["cluster_name", "state", "value", "E_F"]]
    df["E_F"] = pd.to_datetime(df["E_F"], format="%d/%m/%Y %H:%M:%S", infer_datetime_format=True)
    df["date_index"] = (df["E_F"] - pd.datetime(2019, 12, 31)).dt.days
    df = df.astype({"value": int})
    df = df[["cluster_name", "date_index", "value"]]

    # Sort and remove duplicates to obtain max for a given date.
    df.sort_values(
        by=["cluster_name", "date_index", "value"], ascending=[True, True, False], inplace=True
    )
    df.drop_duplicates(["cluster_name", "date_index"], keep="first", inplace=True)
    df["cluster_name"] = df.cluster_name.replace(CHRIS_MAP).str.lower()

    df = df.groupby(["date_index", "cluster_name"]).sum().reset_index()

    return df


if __name__ == "__main__":
    main()