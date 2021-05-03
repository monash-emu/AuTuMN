import pandas as pd
import os
import sqlite3
import boto3


COVID_BASE_DATE = pd.datetime(2019, 12, 31)
DATA_PATH = "M:\Documents\@Projects\Covid_consolidate\output"
phl = ["calabarzon", "central-visayas", "davao-city", "manila", "philippines"]
mys = ["selangor", "penang", "malaysia", "kuala-lumpur", "johor"]
os.chdir(DATA_PATH)

list_of_files = os.listdir(DATA_PATH)
dict_of_files = {city: each  for city in phl + mys for each in list_of_files if city in each}
dict_of_files = {each: os.path.join(DATA_PATH, dict_of_files[each]) for each in dict_of_files}

df_mle = pd.DataFrame()
df_un = pd.DataFrame()
query_do = "SELECT scenario,times,notifications,incidence,icu_occupancy,hospital_occupancy FROM derived_outputs;"
query_un = "SELECT scenario,time,type, value FROM uncertainty WHERE quantile=0.5 AND type in ('incidence','notifications','icu_occupancy','accum_deaths', 'infection_deaths');"
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

col_dict = {
    "hospital_occupancy": "mle_hospital_occupancy",
    "notifications_x": "mle_notifications",
    "incidence_x": "mle_incidence",
    "icu_occupancy_x": "mle_icu_occupancy",
    "accum_deaths": "median__accum_deaths",
    "icu_occupancy_y": "median__icu_occupancy",
    "incidence_y": "median__incidence",
    "notifications_y": "median__notifications",
    "infection_deaths": "median_infection_deaths",
}

df["Date"] = pd.to_timedelta(df.times, unit="days") + (COVID_BASE_DATE)

df.rename(
    columns=col_dict,
    inplace=True,
)


df = df[
    [
        "Region",
        "Date",
        "times",
        "scenario",
        "mle_notifications",
        "mle_incidence",
        "mle_icu_occupancy",
        "mle_hospital_occupancy",
        "median__accum_deaths",
        "median__icu_occupancy",
        "median__incidence",
        "median_infection_deaths",
        "median__notifications",
    ]
]

s3 = boto3.client("s3")
for  k,v in {'mys': mys, 'phl': phl}.items():
    
    df[df.Region.isin(v)].to_csv(f"{k}_data.csv")
    s3.upload_file(f"{k}_data.csv", "autumn-files", f"{k}_data.csv", ExtraArgs={"ACL": "public-read"})
    os.remove(f"{k}_data.csv")


