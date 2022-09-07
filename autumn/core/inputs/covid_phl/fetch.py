"""
This file imports testing data from google drive and saves it to disk as a CSV.
See Readme.md \data\inputs\covid_phl on how to update DATA_URL
"""
import os
from pathlib import Path

from autumn.settings import INPUT_DATA_PATH
from google_drive_downloader import GoogleDriveDownloader as gdd

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

# From DoH google drive folder, shareable link changes with every update
DATA_URL = "1r1IJ11nCrxO0s-G9ugFjqrCCtdZS6H0b"  # shareable link for sheet 07 testing data

COVID_PHL_DIRPATH = INPUT_DATA_PATH / "covid_phl"
COVID_PHL_CSV_PATH = COVID_PHL_DIRPATH / "COVID_Phl_testing.csv"
COVID_PHL_VAC_PATH = COVID_PHL_DIRPATH / "phl_vaccination.csv"


def fetch_covid_phl_data():

    gdd.download_file_from_google_drive(
        file_id=DATA_URL, dest_path=COVID_PHL_CSV_PATH, overwrite=True, showsize=True
    )
    file_size = os.path.getsize(COVID_PHL_CSV_PATH)
    assert (
        file_size > 10000
    ), "File size is too small. Please check PHL google drive link for testing numbers"
