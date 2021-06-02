import pandas as pd
import numpy as np
import os
import sys
import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt


from scripts.phl_data_upload import COVID_BASE_DATETIME
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
start_time = do_df[["scenario", "times"]].groupby(["scenario"]).min().max()[0]

MLE_RUN = mcmc_run_df.sort_values(["accept", "loglikelihood"], ascending=[False, False])[0:1]
MYS_DEATH_URL = "https://docs.google.com/spreadsheets/d/15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo/export?format=xlsx&id=15FGDQdY7Bt2pDD-TVfgKbRAt33UvWdYcdX87IaUXYYo"

MYS_NOTIFICATIONS = os.path.join(INPUT_DATA_PATH,"covid_mys","MALAYSIA_ORIGINAL SHEET_EW52__who-weekly-aggregate-COVID19reportingform.xlsx")



def get_mys_deaths():
    mys_death = pd.read_excel(MYS_DEATH_URL)
    mys_death.rename(
        columns={"State where death occurred": "state", "Age": "age", "Sex": "sex"}, inplace=True
    )
    mys_death = mys_death[["age", "sex", "state"]]
    mys_death.age.replace(["4 Bulan"], [4], inplace=True)
    mys_death.state = mys_death.state.str.strip().str.lower()
    mys_death.dropna(subset=["age"], inplace=True)
    mys_death = mys_death.groupby(by=["age", "state"]).count().reset_index()

    mys_national_death = mys_death.groupby(by=["age"]).count().reset_index()
    mys_national_death["state"] = "malaysia"

    mys_death = mys_death.append(mys_national_death)
    mys_death.rename(columns={"sex": "death"}, inplace=True)
    mys_death = mys_death[mys_death.state == REGION]
    bins = list(range(0, 130, 5))

    groups = mys_death.groupby(pd.cut(mys_death.age, bins)).sum()
    groups.index = bins[:-1]
    groups.age = groups.index
    return groups

def get_do_notif():
    do_notif = pd.concat(pd_list)
    do_notif = do_notif[
        (do_notif.chain == MLE_RUN.chain.values[0]) & (do_notif.run == MLE_RUN.run.values[0])
    ]
    cols = [col for col in do_notif.columns if "notificationsXagegroup_" in col]
    cols = BASE_COL + cols
    do_notif = do_notif[cols]

    do_notif = pd.melt(do_notif, id_vars=BASE_COL, var_name="age", value_name="do_notif")
    do_notif.age = do_notif.age.str.split("X").str.get(1).str.split("_").str.get(1)

    cutoff_date = (pd.to_datetime('today') - COVID_BASE_DATETIME).days
    do_notif = do_notif[do_notif.times < cutoff_date]

    do_notif = do_notif.groupby(by=["scenario", "age"]).sum()
    
    do_notif = do_notif.reset_index()[["scenario", "age", "do_notif"]]
    do_notif = do_notif.astype({'age': 'int32'})

    bins = [-1,4,14,24,34,44,54,64,74,129]

    groups = do_notif.groupby(by=['scenario',pd.cut(do_notif.age, bins)]).sum()
    groups = groups['do_notif'].reset_index()
    groups.age = groups.age.astype(str)
    groups.age = groups.age.replace({'(-1, 4]':'0','(4, 14]':'5', '(14, 24]':'15', '(24, 34]':'25', '(34, 44]':'35',
       '(44, 54]':'45', '(54, 64]':'55', '(64, 74]':'65', '(74, 129]':'75'})
    
    return groups


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



mys_notif = pd.read_excel(MYS_NOTIFICATIONS,header=4, usecols="B,J:T")
mys_notif = pd.melt(mys_notif, id_vars='week_start_date', var_name="age", value_name="notif")
mys_notif.fillna(0, inplace =True)
mys_notif.age = mys_notif.age.str.split("_").str.get(1)
mys_notif.age.replace(['85above'],['75'], inplace=True) # compare with derived outputs 75+
mys_notif = mys_notif.groupby(by='age').sum().reset_index()

do_notif = get_do_notif()

do_notif.merge(mys_notif, how='outer', on='age')



mys_death = get_mys_deaths()
do_notif = get_do_notif()
do_notif = do_notif.merge(mys_death, how='left', on=['age'])
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
for each_scenario in do_notif.scenario.unique():
    
    plt_df = do_notif[do_notif.scenario==each_scenario]
    plt_df = plt_df.sort_values(by=['age'])

    labels = plt_df.age
    deaths = plt_df.death
    modelled_deaths = plt_df.do_death

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, deaths, width, label='Deaths')
    rects2 = ax.bar(x + width/2, modelled_deaths, width, label='Modelled deaths')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Deaths')
    ax.set_title(f'{REGION} deaths by scenario {each_scenario}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
       
    ax.legend()

    fig.tight_layout()
    

plt.show()


