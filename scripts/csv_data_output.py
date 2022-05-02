from datetime import datetime
import os

import sqlite3
import pandas as pd
import boto3

s3 = boto3.client("s3")
COVID_BASE_DATE = datetime(2019, 12, 31)
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
]

AGE_GROUPS = list(range(0, 80, 5))

phl = {
    "region": [
        "calabarzon",
        "central-visayas",
        "davao-city",
        "davao-region",
        "manila",
        "philippines",
    ],
    "columns": STANDARD_COL,
}
mys = {
    "region": ["selangor", "penang", "malaysia", "kuala-lumpur", "johor"],
    "columns": STANDARD_COL,
}

lka = {"region": ["sri_lanka"], "columns": STANDARD_COL}

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
    "infection_deaths",
    "notifications",
]
VIC_REQUEST = [
    f"{output}_for_cluster_{cluster.lower()}" for output in VIC_OUTPUT for cluster in VIC_CLUSTERS
]

vic = {
    "region": ["victoria"],
    "columns": VIC_REQUEST
    + [
        "notifications",
        "hospital_occupancy",
        "icu_occupancy",
        "hospital_admissions",
        "icu_admissions",
        "infection_deaths",
    ],
}

npl_incidence_col = [f"incidenceXagegroup_{each_age}" for each_age in AGE_GROUPS]
npl = {"region": ["nepal"], "columns": STANDARD_COL + npl_incidence_col}


def upload_csv(country_list):
    for ctry in country_list:
        s3.upload_file(
            f"{ctry}_data.csv", "autumn-files", f"{ctry}_data.csv", ExtraArgs={"ACL": "public-read"}
        )
        os.remove(f"{ctry}_data.csv")


def get_files(country):
    return {
        region: os.path.join(DATA_PATH, each)
        for region in country["region"]
        for each in list_of_files
        if region in each
    }


phl["region"] = get_files(phl)
mys["region"] = get_files(mys)
lka["region"] = get_files(lka)
# npl["region"] = get_files(npl)
# vic["region"] = get_files(vic)

country = {
    "lka": lka,
    "phl": phl,
    "mys": mys,
}  # "npl": npl, "vic": vic}

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
    df_un["type"] = df_un["type"] + "_P" + df_un["quantile"].astype(str)
    df_un = pd.pivot_table(
        df_un, values="value", index=["Region", "time", "scenario"], columns=["type"]
    )
    df_un.reset_index(inplace=True)

    df = df_mle.merge(
        df_un,
        how="outer",
        left_on=["Region", "scenario", "times"],
        right_on=["Region", "scenario", "time"],
        suffixes=("_mle", "_un"),
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

upload_csv(["lka", "phl", "mys"])
