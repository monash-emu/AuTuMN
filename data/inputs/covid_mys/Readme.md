Calibration data for Malaysia is obtained from WPRO COVID modelling shared folder.
https://drive.google.com/drive/u/2/folders/1C-OsJbjdhQmpmedBZ8XuM65ANBO22dXc

In this folder, there are two files; a google sheet and an excel file.

Google sheet is titled: `Total Death Covid19 in Malaysia_updated 30 August 2020` (date will vary)
Location:	https://docs.google.com/spreadsheets/d/1cQe1k7GQRKFzcfXXxdL7_pTNQUMpOAPX3PqhPsI4xt8/edit?usp=sharing
Use case: Download and save file as 'COVID_MYS_DEATH.csv'

Excel sheet is titled is: `New _import_ICU cases in Malaysia_updated 30 August 2020` (date will vary)
Location:	https://drive.google.com/file/d/1mnZcmj2jfmrap1ytyg_ErD1DZ7zDptyJ/view?usp=sharing
Use case:	File is created when running MYS update script in ./scripts/mys_data_upload.py

Malaysia regional population stats
https://www.data.gov.my/data/dataset/ec71711f-9b1f-42cd-9229-0c3e1f0e1dbb/resource/423f8f8c-5b74-4b5b-9ba9-d67a5c80e22c/download/mid-year-population-estimates-by-age-group-sex-and-state-malaysia-2015-2019.csv

Data for Sabah is from https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Sabah
Case numbers from New cases per day figure is entered into sabah.csv
Run sabah_data_upload.py in /scripts to update calibration targets.
