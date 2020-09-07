"""
This file imports testing data from google drive and saves it to disk as a CSV.
"""
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from autumn import constants

# From DoH google drive folder, shareable link changes with every update
DATA_URL = "14xQto5vW20ktDacGCB6asmcggKoHm8K0" # shareable link
COVID_PHL_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "covid_phl")
COVID_PHL_CSV_PATH = os.path.join(COVID_PHL_DIRPATH, "COVID_Phl_testing.csv")



def fetch_covid_phl_data():
    gdd.download_file_from_google_drive(file_id = DATA_URL, dest_path = COVID_PHL_CSV_PATH)
