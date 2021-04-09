import pandas as pd
import os
import sqlite3
import boto3


COVID_BASE_DATE = pd.datetime(2019, 12, 31)
DATA_PATH = "M:\Documents\@Projects\Covid_consolidate\output"
phl = ["calabarzon", "central-visayas", "davao-city", "manila", "philippines"]
os.chdir(DATA_PATH)

list_of_files = os.listdir(DATA_PATH)
dict_of_files = {city: each for each in list_of_files for city in phl if city in each}
dict_of_files = {each: os.path.join(DATA_PATH, dict_of_files[each]) for each in dict_of_files}

df = pd.DataFrame()
for region in dict_of_files:
    conn = sqlite3.connect(dict_of_files[region])
    query = "SELECT scenario,times,notifications,incidence,icu_occupancy FROM derived_outputs;"
    if df.empty:
        df = pd.read_sql_query(query, conn)
        df["Region"] = region
    else:
        df_temp = pd.read_sql_query(query, conn)
        df_temp["Region"] = region
        df = df.append(df_temp)

df["Date"] = pd.to_timedelta(df.times, unit="days") + (COVID_BASE_DATE)
df = df[["Region", "Date", "times", "scenario", "notifications", "incidence", "icu_occupancy"]]
df.to_csv("phl_data.csv")


s3 = boto3.client("s3")
s3.upload_file("phl_data.csv", "autumn-files", "phl_data.csv",ExtraArgs={'ACL':'public-read'})
os.remove("phl_data.csv")
