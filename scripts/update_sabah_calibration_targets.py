##!/usr/bin/env python
"""
Script for writing calibration targets for sabah
"""
import os
import sys
import json
from autumn import constants
import pandas as pd


COVID_SABAH_CSV = os.path.join(constants.INPUT_DATA_PATH, "covid_mys", "sabah.csv")
COVID_BASE_DATE = pd.datetime(2019, 12, 31)
REGION_DIR = os.path.join(constants.APPS_PATH, "covid_19", "regions", "sabah")

TARGETS_MAP = {"notifications": "NC"}


def main():

    update_calibration()


def update_calibration():
    """
    Update Sabah's calibration targets
    """

    df = pd.read_csv(COVID_SABAH_CSV)

    file_path = os.path.join(REGION_DIR, "targets.json")
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS_MAP.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)

    return df


if __name__ == "__main__":
    main()

