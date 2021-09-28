import pandas as pd
import os
import sqlite3
import boto3
from datetime import datetime

s3 = boto3.client("s3")
COVID_BASE_DATE = pd.datetime(2019, 12, 31)
DATA_PATH = "M:\Documents\@Projects\Covid_consolidate\output"
os.chdir(DATA_PATH)
list_of_files = os.listdir(DATA_PATH)

STANDARD_COL = [
    "incidence",
    "notifications",
    "hospital_occupancy",
    "icu_occupancy",
    "accum_deaths",
    "infection_deaths",
    "hospital_admissions",
    "icu_admissions",
]

AGE_GROUPS = list(range(0, 80, 5))


VIC_CLUSTERS = [
    "BARWON_SOUTH_WEST",
    "GIPPSLAND",
    "GRAMPIANS",
    "HUME",
    "LODDON_MALLEE",
    "NORTH_METRO",
    "SOUTH_EAST_METRO",
    "SOUTH_METRO",
    "WEST_METRO",
]
VIC_OUTPUT = [
    "hospital_occupancy",
    "icu_occupancy",
    "hospital_admissions",
    "icu_admissions",
    "notifications",
]
VIC_REQUEST = [
    f"{output}Xcluster_{cluster.lower()}" for output in VIC_OUTPUT for cluster in VIC_CLUSTERS
]

vic = {
    "region": ["victoria"],
    "columns": VIC_REQUEST + STANDARD_COL,
}


def get_files(country):
    return {
        region: os.path.join(DATA_PATH, each)
        for region in country["region"]
        for each in list_of_files
        if region in each
    }


def fix_col(df):
    df.rename(columns={"Region": "region", "times": "time"}, inplace=True)
    return df


def process_df(df):
    try:
        df.rename(columns={"times": "time"}, inplace=True)
    except:
        print("no times col")

    df["time"] = pd.to_timedelta(df.time, unit="days") + (COVID_BASE_DATE)

    df[["type", "region"]] = df["type"].str.split("X", expand=True)
    df["region"] = df["region"].str.replace("cluster_", "").str.upper()

    return df


vic["region"] = get_files(vic)

country = {"vic": vic}

for ctry in country:

    df_mle = pd.DataFrame()
    df_un = pd.DataFrame()

    query_do = (
        "SELECT scenario, times, "
        + "".join({each + ", " for each in country[ctry]["columns"]})[:-2]
        + " FROM derived_outputs;"
    )
    query_un = (
        "SELECT scenario,time,type,quantile, value FROM uncertainty WHERE type in ("
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

    id_col = ["Region", "scenario", "times"]
    val_col = [each for each in list(df_mle.columns) if each not in id_col]

    df_mle = df_mle.melt(id_vars=id_col, value_vars=val_col, var_name="type", value_name="value")

    df_mle = fix_col(df_mle)
    df_mle = process_df(df_mle)

    df_un = fix_col(df_un)
    df_un = process_df(df_un)

    df = df_mle.append(df_un)

    df["region"] = df.region.fillna("VICTORIA")

    date_val = [file for file in list_of_files if "victoria" in file][0].split("-")[3]
    date_val = datetime.fromtimestamp(int(date_val))
    commit = [file for file in list_of_files if "victoria" in file][0].split("-")[4].split(".")[0]
    date_val = str(date_val).replace(":", "-").replace(" ", "T")

    file_name = f"vic-forecast-{commit}-{date_val}.csv"
    df.to_csv(file_name)
