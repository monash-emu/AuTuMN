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

df_mle = pd.DataFrame()
df_un = pd.DataFrame()
query_do = "SELECT scenario,times,notifications,incidence,icu_occupancy,hospital_occupancy FROM derived_outputs;"
query_un = "SELECT scenario,time,type, value FROM uncertainty WHERE quantile=0.5 AND type in ('incidence','notifications','icu_occupancy','accum_deaths');"
for region in dict_of_files:
    conn = sqlite3.connect(dict_of_files[region])
    if df_mle.empty:
        df_mle = pd.read_sql_query(query_do, conn)
        df_mle["Region"] = region
    else:
        df_temp = pd.read_sql_query(query_do, conn)
        df_temp["Region"] = region
        df_mle = df_mle.append(df_temp)
    if df_un.empty:
        df_un = pd.read_sql_query(query_un, conn)
        df_un["Region"] = region
    else:
        df_temp = pd.read_sql_query(query_un, conn)
        df_temp["Region"] = region
        df_un = df_un.append(df_temp)

df_un = pd.pivot_table(
    df_un, values="value", index=["Region", "time", "scenario"], columns=["type"]
)
df_un.reset_index(inplace=True)
df = df_mle.merge(
    df_un,
    how="outer",
    left_on=["Region", "scenario", "times"],
    right_on=["Region", "scenario", "time"],
)


df["Date"] = pd.to_timedelta(df.times, unit="days") + (COVID_BASE_DATE)
df = df[
    [
        "Region",
        "Date",
        "times",
        "scenario",
        "notifications_x",
        "incidence_x",
        "icu_occupancy_x",
        "accum_deaths",
        "icu_occupancy_y",
        "incidence_y",
        "notifications_y",
    ]
]
df.rename(
    columns={
        "notifications_x": "mle_notifications_x",
        "incidence_x": "mle_incidence",
        "icu_occupancy_x": "mle_icu_occupancy",
        "accum_deaths": "un_accum_deaths",
        "icu_occupancy_y": "un_icu_occupancy",
        "incidence_y": "un_incidence",
        "notifications_y": "un_notifications",
    },
    inplace=True,
)
df.to_csv("phl_data.csv")


s3 = boto3.client("s3")
s3.upload_file("phl_data.csv", "autumn-files", "phl_data.csv", ExtraArgs={"ACL": "public-read"})
os.remove("phl_data.csv")
