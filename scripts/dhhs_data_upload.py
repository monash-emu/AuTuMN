#!/usr/bin/env python
"""
Script for loading DHHS data into calibration targets and import inputs.
"""
import os
import sys
import json
from datetime import datetime
from getpass import getpass

import pandas as pd

# Do some super sick path hacks to get script to work from command line.
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from autumn import constants
from autumn import secrets


DHHS_CSV = os.path.join(constants.INPUT_DATA_PATH, "monashmodelextract.secret.csv")
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
    for region in CLUSTER_MAP.keys():
        current_cluster = CLUSTER_MAP[region].lower()
        update_df = cal_df[cal_df.cluster_name == current_cluster]
        file_path = os.path.join(REGION_DIR, current_cluster, "targets.secret.json")
        with open(file_path, mode="r") as f:
            targets = json.load(f)

        for key, val in TARGETS_MAP.items():
            targets[key]["times"] = list(update_df["date_index"])
            targets[key]["values"] = list(update_df[val])

        with open(file_path, "w") as f:
            json.dump(targets, f, indent=2)

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
    df["date_index"] = (df.date - pd.datetime(2019, 12, 31)).dt.days

    # Remove last date due to poor accuracy of data.
    df[df.date_index != df.date_index.max()]
         
    return df


if __name__ == "__main__":
    main()