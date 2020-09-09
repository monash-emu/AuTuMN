"""
This file imports testing data from google drive and saves it to disk as a CSV.
See Readme.md \data\inputs\covid_phl on how to update DATA_URL
"""
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from autumn import constants

# From DoH google drive folder, shareable link changes with every update
DATA_URL = "1GE-uO9kaFBgwreu7zFdXhYvG3U_9EY8C" # shareable link
COVID_PHL_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "covid_phl")
COVID_PHL_CSV_PATH = os.path.join(COVID_PHL_DIRPATH, "COVID_Phl_testing.csv")



def fetch_covid_phl_data():
    try:
        os.remove(COVID_PHL_CSV_PATH)
    except:
        print("File not found, unable to delete.")
    gdd.download_file_from_google_drive(file_id = DATA_URL, dest_path = COVID_PHL_CSV_PATH)
