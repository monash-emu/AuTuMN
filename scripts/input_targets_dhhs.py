import os
import pandas as pd

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.tools.utils.utils import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index
from autumn.settings import PASSWORD_ENVAR
from getpass import getpass


# Use OWID csv for notification and death numbers.
COVID_DHHS_TARGETS = os.path.join(
    PROJECTS_PATH, "covid_19", "victoria", "victoria_2021", "targets.secret.json"
)

COVID_DHHS_VAC_CSV = os.path.join(
    INPUT_DATA_PATH, "covid_au", "vac_by_week_final_2021-09-02.csv"
)  # TODO - parse for latest file, not hardcode.
COVID_DHHS_CASE_CSV = os.path.join(
    INPUT_DATA_PATH, "covid_au", "NCOV_COVID_Cases_by_LGA_Source_20210903.csv"
)
COVID_VIC_POSTCODE_POP_CSV = os.path.join(
    INPUT_DATA_PATH, "covid_au", "COVID19 Data Viz Postcode data - postcode.csv"
)
COVID_DHHS_CLUSTERS_CSV = os.path.join(
    INPUT_DATA_PATH, "mobility", "LGA to Cluster mapping dictionary with proportions.csv"
)


CLUSTER_MAP = {
    1: "NORTH_METRO",
    2: "SOUTH_EAST_METRO",
    3: "SOUTH_METRO",
    4: "WEST_METRO",
    5: "BARWON_SOUTH_WEST",
    6: "GIPPSLAND",
    7: "GRAMPIANS",
    8: "HUME",
    9: "LODDON_MALLEE",
    0: "VIC",
}


def preprocess_postcode_pop():
    df = pd.read_csv(COVID_VIC_POSTCODE_POP_CSV)
    df = df[["postcode", "population"]]

    df = df.groupby(["postcode", "population"]).size().reset_index()

    return df


def preprocess_cases():
    df = pd.read_csv(COVID_DHHS_CASE_CSV)
    df = create_date_index(COVID_BASE_DATETIME, df, "diagnosis_date")
    df.groupby

    df = df.groupby(["date_index", "localgovernmentarea"]).size().reset_index()
    df.replace({"Melton (C)": "Melton (S)", "Wodonga (C)": "Wodonga (RC)"}, inplace=True)

    df.rename(columns={0: "cases"}, inplace=True)

    return df


cluster_map_df = pd.read_csv(COVID_DHHS_CLUSTERS_CSV)
cases_df = preprocess_cases()

cases_df = cases_df.merge(
    cluster_map_df, left_on=["localgovernmentarea"], right_on=["lga_name"], how="left"
)
cases_df["cluster_cases"] = cases_df.cases * cases_df.proportion
cases_df.loc[cases_df.cluster_id.isna(), "cluster_id"] = "VIC"

cases_df = (
    cases_df[["date_index", "cluster_id", "cluster_cases"]]
    .groupby(["date_index", "cluster_id"])
    .sum()
    .reset_index()
)
cases_df.cluster_id.replace(CLUSTER_MAP, inplace=True)


for cluster in CLUSTER_MAP.values():

    TARGET_MAP_DHHS = {
    f"notifications_for_cluster_{cluster.lower()}": "cluster_cases"}

    cluster_df = cases_df.loc[cases_df.cluster_id==cluster]
    password = os.environ.get(PASSWORD_ENVAR, "") 
    if not password:
        password = getpass(prompt="Enter the encryption password:")
    update_timeseries(TARGET_MAP_DHHS,cluster_df,COVID_DHHS_TARGETS,password)


