'''add comment in script explaining what its for
This is where the scripts to prepross the data go
save files in data/targets/
'''
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
import itertools
from google_drive_downloader import GoogleDriveDownloader as gdd
import json

# shareable google drive links
PHL_doh_link = '1U0iNYB9ZEMTYRMp9Eh1yYOocJ9rbLsXI'
PHL_fassster_link = '1JLi6uTbUIe1DNNI_cjT5QysPhcSnq5_2'

# destination folders filepaths
base_dir = os.path.abspath(os.curdir) 
#BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
#sys.path.append(BASE_DIR)
PHL_doh_dest = './data/targets/PHL_icu.csv'
PHL_fassster_dest = './data/targets/PHL_ConfirmedCases.zip'
icu_dest = './data/targets/PHL_icu_processed.csv'
deaths_dest = './data/targets/PHL_deaths_processed.csv'
notifications_dest = './data/targets/PHL_notifications_processed.csv'
targets_dir = './data/targets/'

# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

# function to fetch data
def fetch_phl_data():
    gdd.download_file_from_google_drive(file_id = PHL_doh_link, dest_path = PHL_doh_dest)
    gdd.download_file_from_google_drive(file_id = PHL_fassster_link, dest_path = PHL_fassster_dest, unzip = True)
    os.remove(PHL_fassster_dest) # remove zip folder

# function to preprocess Philippines data for calibration targets
def process_phl_data():
    # read csvs
    up_to_data_fassster_filename = [filename for filename in os.listdir(targets_dir) if filename.startswith("ConfirmedCases_Final_")]
    up_to_data_fassster_filename = targets_dir + str(up_to_data_fassster_filename[-1])
    fassster_data = pd.read_csv(up_to_data_fassster_filename)
    doh_data = pd.read_csv(PHL_doh_dest)
    # rename regions
    doh_data['region'] = doh_data['region'].replace({'NATIONAL CAPITAL REGION (NCR)': 'manila',
                                                     'REGION IV-A (CALABAR ZON)': 'calabarzon', 
                                                     'REGION VII (CENTRAL VISAYAS)': 'central-visayas'})
    fassster_data['Region'] = fassster_data['Region'].replace({'NCR': 'manila',
                                                     '4A': 'calabarzon', 
                                                     '07': 'central-visayas'})
    # duplicate data to create 'philippines' region and join with original dataset
    doh_data_dup = doh_data.copy()
    fassster_data_dup = fassster_data.copy()    
    doh_data_dup['region'] = 'philippines'
    fassster_data_dup ['Region'] = 'philippines'    
    doh_data = doh_data.append(doh_data_dup)
    fassster_data = fassster_data.append(fassster_data_dup)
    # filter by regions (exclude all regions not modeled) 
    regions = ['calabarzon', 'central-visayas', 'manila', 'philippines']
    doh_data = doh_data[doh_data['region'].isin(regions)]
    fassster_data = fassster_data[fassster_data['Region'].isin(regions)]
    ## most recent ICU data
    doh_data['reportdate'] = pd.to_datetime(doh_data['reportdate'])
    doh_data['times'] = doh_data.reportdate - COVID_BASE_DATETIME
    doh_data['times'] = doh_data['times'] / np.timedelta64(1, 'D')
    icu_occ_at_maxDate =  doh_data.groupby(['region'], as_index = False)['times', 'icu_o'].max()
    icu_occ_at_maxDate.to_csv(icu_dest)
    ## accumulated deaths
    fassster_data = fassster_data[fassster_data['Date_Died'].notna()]
    fassster_data['Date_Died'] = pd.to_datetime(fassster_data['Date_Died'])
    fassster_data['times'] = fassster_data.Date_Died - COVID_BASE_DATETIME
    fassster_data['times'] = fassster_data['times'] / np.timedelta64(1, 'D')
    fassster_data['value'] = 1
    cumulative_deaths_max_date = fassster_data.groupby(['Region'], as_index = False)['times'].max()
    cumulative_deaths_agg = fassster_data.groupby(['Region'], as_index = False)['value'].sum()
    cumulative_deaths = cumulative_deaths_max_date.join(cumulative_deaths_agg.set_index('Region'), on = 'Region')
    cumulative_deaths.to_csv(deaths_dest)
    ## notifications
    fassster_data['imputed_Date_Admitted'] = pd.to_datetime(fassster_data['imputed_Date_Admitted'])
    # make sure all dates within range are included
    dateIndex = pd.date_range(min(fassster_data['imputed_Date_Admitted']), max(fassster_data['imputed_Date_Admitted']))
    all_regions_x_dates = pd.DataFrame(list(itertools.product(regions, dateIndex)), columns = ['Region', 'imputed_Date_Admitted'])
    all_regions_x_dates['imputed_Date_Admitted'] = pd.to_datetime(all_regions_x_dates['imputed_Date_Admitted'])
    fassster_data['times'] = fassster_data.imputed_Date_Admitted - COVID_BASE_DATETIME
    fassster_data['times'] = fassster_data['times'] / np.timedelta64(1, 'D')
    # give each case a value of 1 and a value of 0 for rows added for missing dates
    fassster_data.loc[fassster_data['times'].isna() == True, 'daily_notifications'] = 0
    fassster_data.loc[fassster_data['times'].isna() == False, 'daily_notifications'] = 1
    # sum values by day and region
    fassster_data_agg = fassster_data.groupby(['Region', 'times'], as_index = False)['daily_notifications'].sum()
    # calculate a 7-day rolling window value
    fassster_data_agg = fassster_data_agg.sort_values(['Region', 'times'], ascending=[True, True])
    fassster_data_agg['mean_daily_notifications'] = fassster_data_agg.groupby('Region').rolling(7)['daily_notifications'].mean().reset_index(0, drop = True)
    fassster_data_agg['mean_daily_notifications'] = np.round(fassster_data_agg['daily_notifications'])
    fassster_data_final = fassster_data_agg[['Region', 'times', 'mean_daily_notifications']]
    fassster_data_final = fassster_data_final[fassster_data_final.times > 60]
    fassster_data_final.to_csv(notifications_dest)
    # remove pre-processed files
    os.remove(up_to_data_fassster_filename)
    os.remove(PHL_doh_dest)

phl_regions = ['calabarzon', 'central_visayas', 'manila', 'philippines']

def update_calibration_phl():
    # read in csvs
    icu = pd.read_csv(icu_dest)
    deaths = pd.read_csv(deaths_dest)
    notifications = pd.read_csv(notifications_dest)
    for region in phl_regions:
        icu_tmp = icu.loc[icu['region'] == region]
        deaths_tmp = deaths.loc[deaths['Region'] == region]
        notifications_tmp = notifications.loc[notifications['Region'] == region]
        file_path = os.path.join(base_dir + '\\apps\\covid_19\\regions\\' + region + '\\targets.json')
        
        with open(file_path, mode="r") as f:
            targets = json.load(f)

            targets['notifications']['times'] = list(notifications_tmp['times'])
            targets['notifications']['values'] = list(notifications_tmp['mean_daily_notifications'])
            targets['icu_occupancy']['times'] = list(icu_tmp['times'])
            targets['icu_occupancy']['values'] = list(icu_tmp['icu_o'])
            targets['infection_deaths']['times'] = list(deaths_tmp['times'])
            targets['infection_deaths']['values'] = list(deaths_tmp['value'])

        with open(file_path, "w") as f:
            json.dump(targets, f, indent=2)

