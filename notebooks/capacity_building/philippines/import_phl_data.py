# Python standard library imports come first
from datetime import datetime, timedelta  # We use datetime to manipulate date-time indexes
import requests # To download google drive files
import io # To download google drive zip file
from zipfile import ZipFile # To manage zip files

# Then external package imports
import pandas as pd  # pd is an alias for pandas. This is similar to dataframes in R
import numpy as np

# By defining a region variable, we can easily change the analysis later.
REGION = "NATIONAL CAPITAL REGION (NCR)"

"""
Utility functions
"""

def fetch_doh_data(link:str)->pd.DataFrame:
    """Requests for the DoH 05 cases from the google drive data repository"""
    doh = f"https://drive.google.com/uc?id={link}&export=download&confirm=t"
    df = pd.read_csv(doh)

    return df

def get_fassster_data(link:str)->pd.DataFrame:
    """Reads a google drive zip file and extracts the data from it in memory"""


    faster = f"https://drive.google.com/uc?id={link}&export=download&confirm=t"
    req = requests.get(faster)
    file_like_object = io.BytesIO(req.content)
    zipfile_ob = ZipFile(file_like_object)
    filename = [
        each for each in zipfile_ob.namelist() if each.startswith("2022")
    ]
    df = pd.read_csv(zipfile_ob.open(filename[0]))
    return df

def get_population_and_epi_data(phl_doh_link: str, phl_fasster_link: str) -> (int, pd.DataFrame):
    """
    Master function that loads the Philippines population data and COVID-19 timeseries

    Args:
        phl_doh_link: Google Drive URL to DoH data
        phl_fasster_link: Google Drive URL to FASSSTER data 

    Returns:
        initial_population: The size of the NCR population 
        df: Epidemic data stored in a pandas dataframe
    """
    doh_df = fetch_doh_data(phl_doh_link)
    df_cases = get_fassster_data(phl_fasster_link)

    df_cases = df_cases.groupby(["Report_Date","Region"],as_index=False).size() # Because each row is a case we can use the group size.
    df_cases = df_cases[df_cases['Region']=="NCR"] # Filter for NCR cases
    df_cases = df_cases.rename(columns={"Report_Date":"reportdate", "size":"cases"}) # Rename columns to match DoH names.
    df_cases['reportdate'] = pd.to_datetime(df_cases['reportdate'])

    doh_df = doh_df[['reportdate','region','cfname','nonicu_o','icu_o']]
    doh_df["reportdate"]  = pd.to_datetime(doh_df["reportdate"]).dt.tz_localize(None)
    doh_df = doh_df.groupby(['reportdate','region'], as_index=False).sum()

    mask = (doh_df['region'] == REGION)
    doh_df = doh_df[mask]


    df = pd.merge(doh_df,df_cases, how='left', on='reportdate')
    # Set the index of this DataFrame to use calendar dates
    df = df.set_index('reportdate')


    population_url = 'https://github.com/monash-emu/AuTuMN/raw/master/data/inputs/world-population/subregions.csv'
    df_pop = pd.read_csv(population_url)
    df_pop = df_pop[df_pop['region']=='Metro Manila']

    df_pop = df_pop.melt(id_vars=['country','iso3','region','year'], var_name='age_group', value_name='pop')
    df_pop = df_pop[['region','pop']].groupby('region',as_index=False).sum()
    initial_population = df_pop['pop'][0] * 1000 # We need to multiply by 1000 to covert it back to counts.

    return initial_population, df