import pandas as pd
import os
import sqlite3
import boto3

s3 = boto3.client("s3")
COVID_BASE_DATE = pd.datetime(2019, 12, 31)
DATA_PATH = "M:\Documents\@Projects\Covid_consolidate\output"
phl = {
    "region": ["calabarzon", "central-visayas", "davao-city", "manila", "philippines"],
    "columns": [
        "incidence",
        "notifications",
        "icu_occupancy",
        "accum_deaths",
        "accum_incidence",
        "accum_notifications",
        "infection_deaths",
    ],
}
mys = {
    "region": ["selangor", "penang", "malaysia", "kuala-lumpur", "johor"],
    "columns": [
        "incidence",
        "notifications",
        "hospital_occupancy",
        "icu_occupancy",
        "accum_deaths",
        "infection_deaths",
    ],
}
os.chdir(DATA_PATH)


list_of_files = os.listdir(DATA_PATH)

phl["region"] = {
    region: os.path.join(DATA_PATH, each)
    for region in phl["region"]
    for each in list_of_files
    if region in each
}
mys["region"] = {
    region: os.path.join(DATA_PATH, each)
    for region in mys["region"]
    for each in list_of_files
    if region in each
}

country = {"phl": phl, "mys": mys}

for ctry in country:

    df_mle = pd.DataFrame()
    df_un = pd.DataFrame()

    query_do = (
        "SELECT scenario, times, "
        + "".join({each + ", " for each in country[ctry]["columns"]})[:-2]
        + " FROM derived_outputs;"
    )
    query_un = (
        "SELECT scenario,time,type, value FROM uncertainty WHERE quantile=0.5 AND type in ("
        + "".join({"'" + each + "', " for each in country[ctry]["columns"]})[:-2]
        + ");"
    )

    for app_name in country[ctry]["region"]:
        reg_file = country[ctry]["region"][app_name]

        conn = sqlite3.connect(reg_file)
        if df_mle.empty:
            df_mle = pd.read_sql_query(query_do, conn)
            df_mle["Region"] = app_name
        else:
            df_temp = pd.read_sql_query(query_do, conn)
            df_temp["Region"] = app_name
            df_mle = df_mle.append(df_temp)
        if df_un.empty:
            df_un = pd.read_sql_query(query_un, conn)
            df_un["Region"] = app_name
        else:
            df_temp = pd.read_sql_query(query_un, conn)
            df_temp["Region"] = app_name
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
        suffixes=("_mle", "_median"),
    )

    df["Date"] = pd.to_timedelta(df.times, unit="days") + (COVID_BASE_DATE)

    df.rename(
        columns={
            "hospital_occupancy": "hospital_occupancy_mle",
            "infection_deaths": "infection_deaths_mle",
        },
        inplace=True,
    )

    col_set1 = ["Region", "scenario", "Date", "times", "time"]
    col_set2 = [col for col in list(df.columns) if col not in col_set1]
    col_set2.sort()

    col_set1 = col_set1[:-1]
    df = df[col_set1 + col_set2]

    df.to_csv(f"{ctry}_data.csv")
    s3.upload_file(
        f"{ctry}_data.csv", "autumn-files", f"{ctry}_data.csv", ExtraArgs={"ACL": "public-read"}
    )
    os.remove(f"{ctry}_data.csv")
