from autumn.tools.utils.utils import COVID_BASE_DATETIME
import pandas as pd
import numpy as np
import os
import sys
import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt

# Don't ask me why sigh ...
sys.path.append("C:\\Users\\maba0001\\AuTuMN")
from settings import OUTPUT_DATA_PATH

COVID_BASE_DATE = pd.datetime(2019, 12, 31)
RUN_ID = "covid_19/malaysia/1622599229/e9de2f6"
RUN_ID = RUN_ID.split(sep="/")
REGION = [
    region
    for region in {"malaysia", "johor", "selangor", "kuala_lumpur", "penang"}
    if region in RUN_ID
][0]

ts = datetime.datetime.fromtimestamp(int(RUN_ID[2]))
RUN_ID[2] = ts.strftime("%Y-%m-%d")

FEATHER_PATH = os.path.join(OUTPUT_DATA_PATH, "full", RUN_ID[0], RUN_ID[1], RUN_ID[2])
FEATHER_PATH = [x[0] for x in os.walk(FEATHER_PATH)][2:]

DERIVED_OUTPUT = [os.path.join(each, "derived_outputs.feather") for each in FEATHER_PATH]
MCMC_PARAM = [os.path.join(each, "mcmc_params.feather") for each in FEATHER_PATH]
MCMC_RUN = [os.path.join(each, "mcmc_run.feather") for each in FEATHER_PATH]

BASE_COL = ["chain", "run", "scenario", "times"]
REQ_COL = ["incidence", "notifications"]
TIME = 180  # number of days to look forward after the intervention start

pd_list = [pd.read_feather(each) for each in DERIVED_OUTPUT]
mcmc_param_list = [pd.read_feather(each) for each in MCMC_PARAM]
mcmc_run_list = [pd.read_feather(each) for each in MCMC_RUN]

do_df = pd.concat(pd_list)
mcmc_param_df = pd.concat(mcmc_param_list)
params = mcmc_param_df.name.unique()
mcmc_run_df = pd.concat(mcmc_run_list)
do_df = do_df[BASE_COL + REQ_COL]
start_time = do_df[["scenario", "times"]].groupby(["scenario"]).min().max()[0]

MLE_RUN = mcmc_run_df.sort_values(["accept", "loglikelihood"], ascending=[False, False])[0:1]


MYS_DEATH_URL = "https://docs.google.com/spreadsheets/d/15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo/export?format=xlsx&id=15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo"

def perform_calculation(df):
    df_baseline = df.loc[
        df.scenario == 0,
    ]

    df = df.merge(
        df_baseline,
        how="inner",
        left_on=["chain", "run", "times"],
        right_on=["chain", "run", "times"],
        suffixes=("_run", "_baseline"),
    )

    perform_cal_col = [(f"{each}_rel", f"{each}_run", f"{each}_baseline") for each in REQ_COL]

    for col_pair in perform_cal_col:
        df[f"{col_pair[0]}"] = df[col_pair[1]] - df[col_pair[2]]
        df.drop([col_pair[2]], axis=1, inplace=True)

    df.drop(["scenario_baseline"], axis=1, inplace=True)

    return df.loc[
        df.times == (start_time + TIME),
    ]


do_df = perform_calculation(do_df)


mcmc_param_df = mcmc_param_df.pivot_table(
    index=["chain", "run"], columns="name", values="value"
).reset_index()

do_df = do_df.merge(
    mcmc_param_df,
    how="left",
    left_on=["chain", "run"],
    right_on=["chain", "run"],
    suffixes=("_run", "_baseline"),
)

do_df = do_df.merge(mcmc_run_df, how="left", left_on=["chain", "run"], right_on=["chain", "run"])
count_row = mcmc_param_df.shape[0]  # Gives number of rows

