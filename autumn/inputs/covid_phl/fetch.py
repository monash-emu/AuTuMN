"""
This file imports testing data from google drive and saves it to disk as a CSV.
See Readme.md \data\inputs\covid_phl on how to update DATA_URL
"""
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from settings import INPUT_DATA_PATH

# From DoH google drive folder, shareable link changes with every update
DATA_URL = "16_IEN_9p0C1y-pPJL2GyVCyJC5dUxq8j"  # shareable link for sheet 07 testing data
COVID_PHL_DIRPATH = os.path.join(INPUT_DATA_PATH, "covid_phl")
COVID_PHL_CSV_PATH = os.path.join(COVID_PHL_DIRPATH, "COVID_Phl_testing.csv")


def fetch_covid_phl_data():
    try:
        os.remove(COVID_PHL_CSV_PATH)
    except:
        print("File not found, unable to delete.")
    gdd.download_file_from_google_drive(file_id=DATA_URL, dest_path=COVID_PHL_CSV_PATH)
