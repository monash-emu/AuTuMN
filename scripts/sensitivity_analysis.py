from numpy.core.arrayprint import _guarded_repr_or_str
from numpy.lib.shape_base import column_stack
import pandas as pd
import numpy as np
import os
import sys
import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt

from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.settings import OUTPUT_DATA_PATH
from autumn.settings import INPUT_DATA_PATH

RUN_ID = "covid_19/malaysia/1621579054/07755e9"
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
intervention_start = do_df[["scenario", "times"]].groupby(["scenario"]).min().max()[0]

# Establish model start time
model_start = do_df[["times"]].min()[0]

# Establish last date
deaths_cutoff_date = (pd.to_datetime("today") - COVID_BASE_DATETIME).days
notification_cutoff_date = 365

MLE_RUN = mcmc_run_df.sort_values(["accept", "loglikelihood"], ascending=[False, False])[0:1]
MYS_DEATH_URL = "https://docs.google.com/spreadsheets/d/15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo/export?format=xlsx&id=15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo"

MYS_NOTIFICATIONS = os.path.join(
    INPUT_DATA_PATH,
    "covid_mys",
    "MALAYSIA_ORIGINAL SHEET_EW52__who-weekly-aggregate-COVID19reportingform.xlsx",
)


def get_date_range(df, cutoff_date):

    df.times = pd.to_datetime(
        df.times, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    df["date_index"] = (df.times - COVID_BASE_DATETIME).dt.days

    # Filter for start and end
    df = df[(df.date_index >= model_start) & (df.date_index <= cutoff_date)]
    return df


def get_mys_deaths():
    df = pd.read_excel(MYS_DEATH_URL)
    df.rename(
        columns={
            "State where death occurred": "state",
            "Age": "age",
            "Sex": "sex",
            "Date case reported to media": "times",
        },
        inplace=True,
    )

    df = df[["times", "age", "sex", "state"]]
    df = get_date_range(df, deaths_cutoff_date)

    df.age.replace(["4 Bulan"], [4], inplace=True)
    df.state = df.state.str.strip().str.lower()
    df.dropna(subset=["age"], inplace=True)
    df = df.groupby(by=["age", "state"]).count().reset_index()

    mys_national_death = df.groupby(by=["age"]).count().reset_index()
    mys_national_death["state"] = "malaysia"

    df = df.append(mys_national_death)
    df.rename(columns={"sex": "death"}, inplace=True)
    df = df[df.state == REGION]
    bins = list(range(-1, 130, 5))  # -1 to ensure the bins intervals are correct

    groups = df.groupby(pd.cut(df.age, bins)).sum()
    groups.index = [each + 1 for each in bins[:-1]]
    groups.age = groups.index

    return groups[["age", "death"]]


def get_mys_notif():

    df = pd.read_excel(MYS_NOTIFICATIONS, header=4, usecols="B,J:T")
    df.rename(columns={"week_start_date": "times"}, inplace=True)
    df = pd.melt(df, id_vars="times", var_name="age", value_name="notification")
    df.fillna(0, inplace=True)
    df = get_date_range(df, notification_cutoff_date)
    df.age = df.age.str.split("_").str.get(1)
    df.age.replace(
        {"85above": "75", "unk": np.nan}, inplace=True
    )  # compare with derived outputs 75+
    df = df.groupby(by="age").sum().reset_index()
    df = df.astype({"age": "int32"})
    return df


def get_do_feature(feature, cutoff_date):

    do_df = pd.concat(pd_list)
    do_df = do_df[
        (do_df.chain == MLE_RUN.chain.values[0])
        & (do_df.run == MLE_RUN.run.values[0])
        & (do_df.scenario == 0)
    ]

    feature_map = {"notification": "notificationsXagegroup_", "death": "infection_deathsXagegroup_"}

    cols = [col for col in do_df.columns if feature_map[feature] in col]
    cols = BASE_COL + cols
    do_df = do_df[cols]
    do_df = do_df[(do_df.times >= model_start) & (do_df.times <= cutoff_date)]

    do_df = pd.melt(do_df, id_vars=BASE_COL, var_name="age", value_name=f"do_{feature}")
    do_df.age = do_df.age.str.split("X").str.get(1).str.split("_").str.get(1)

    do_df = do_df.groupby(by=["age"]).sum()

    do_df = do_df.reset_index()[["age", f"do_{feature}"]]
    do_df = do_df.astype({"age": "int32"})

    return do_df


def create_sensitivity_df(df):
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
        df.times == (intervention_start + TIME),
    ]


def group_by_age(df, bin):

    df = df.groupby(by=[pd.cut(df.age, bin)]).sum()
    df.drop(columns="age", inplace=True)
    df = df.reset_index()

    df.age = df.age.astype(str)
    df.age = df.age.replace(
        {
            "(-1, 4]": 0,
            "(4, 9]": 5,
            "(9, 14]": 10,
            "(14, 19]": 15,
            "(19, 24]": 20,
            "(24, 29]": 25,
            "(29, 34]": 30,
            "(34, 39]": 35,
            "(39, 44]": 40,
            "(44, 49]": 45,
            "(49, 54]": 50,
            "(54, 59]": 55,
            "(59, 64]": 60,
            "(64, 69]": 65,
            "(69, 74]": 70,
            "(74, 129]": 75,
            # These map the non-standard notification age groups.
            "(4, 14]": 5,
            "(14, 24]": 15,
            "(24, 34]": 25,
            "(34, 44]": 35,
            "(44, 54]": 45,
            "(54, 64]": 55,
            "(64, 74]": 65,
        }
    )

    return df


def plot_do_feature(plt_df, feature):

    plt_df = plt_df.sort_values(by=["age"])

    labels = plt_df.age
    local = plt_df[f"{feature}"]
    modelled = plt_df[f"do_{feature}"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, local, width, label=f"{feature}")
    rects2 = ax.bar(x + width / 2, modelled, width, label=f"Modelled {feature}")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f"{feature}")
    ax.set_title(f"{REGION} {feature} by age")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    fig.tight_layout()
    plt.show()


do_deaths_df = get_do_feature("death", deaths_cutoff_date)
mys_death = get_mys_deaths()

death_bin = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 129]
mys_death = group_by_age(mys_death, death_bin)

do_deaths_df = do_deaths_df.merge(mys_death, how="left", on="age")
plot_do_feature(do_deaths_df, "death")


mys_notif = get_mys_notif()
do_notif_df = get_do_feature("notification", notification_cutoff_date)
notif_bin = [-1, 4, 14, 24, 34, 44, 54, 64, 74, 129]

do_notif_df = group_by_age(do_notif_df, notif_bin)

do_notif_df = do_notif_df.merge(mys_notif, how="left", on="age")
plot_do_feature(do_notif_df, "notification")


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


# Not ideal, will fix
# for each_param in params:
#     for output in REQ_COL:

#         output = f"{output}_rel"
#         fig, ax = plt.subplots()
#         ax.scatter(do_df[each_param], do_df[output], c=do_df["chain"], alpha=0.5)

#         ax.set_xlabel(f"{each_param}", fontsize=15)
#         ax.set_ylabel(f"{output}", fontsize=15)
#         ax.set_title(f"{output} by {each_param}")

#         ax.grid(True)
#         fig.tight_layout()

# plt.show()


# Got here finally


#
# df = get_do_feature()
#
# df.merge(mys_notif, how="outer", on="age")
#
#
# df = get_do_feature()
# df = df.merge(mys_death, how="left", on=["age"])
# do_df = create_sensitivity_df(do_df)