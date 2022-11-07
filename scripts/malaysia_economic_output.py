import pandas as pd
import datetime
import warnings


from autumn.core.plots.utils import REF_DATE
from autumn.core.runs.managed import ManagedRun

from autumn.core import db
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

SCENARIOS = [0, 1, 2, 3, 4, 5, 6]
BASE_COLS = ["year", "scenario", "chain", "run"]
EXTRA_COLS = ["n_immune_none", "n_immune_low", "n_immune_high"]

run_id = "sm_sir/malaysia/1666178263/9dd3ece"
region = "malaysia"

proper_path = Path.cwd() / "data/outputs/runs" / run_id / "data/full_model_runs"
database_paths = db.load.find_db_paths(proper_path)


def preprocess_db(db_path):
    source_db = db.database.get_database(db_path)
    table_df = source_db.query("derived_outputs")
    table_df = table_df[table_df["times"] >= 457]  # After 1st April 2021
    table_df["date"] = table_df["times"].apply(datetime.timedelta) + pd.to_datetime(REF_DATE)
    table_df["month"] = table_df["date"].dt.month
    table_df["year"] = table_df["date"].dt.year
    return table_df


def get_full_derived_outputs():
    for db_path in database_paths:
        table_df = preprocess_db(db_path)
        yield table_df.groupby(BASE_COLS, as_index=False).sum()


def get_mle_outputs(chain, run):
    for db_path in database_paths:
        table_df = preprocess_db(db_path)
        mask = (table_df["chain"] == chain) & (table_df["run"] == run)
        table_df = table_df[mask]
        yield table_df.groupby(BASE_COLS + ["month"], as_index=False).sum()


def output_files(file_type, cols_to_output, df):
    df = df[cols_to_output]
    for scenario in SCENARIOS:
        required_outputs = df.loc[(df["scenario"] == scenario)]
        required_outputs.to_csv(f"{Path.cwd()}/{file_type}_{scenario}.csv", index=False)


df = pd.concat(get_full_derived_outputs())
df = df.sort_values(by=BASE_COLS)
cols_to_output = BASE_COLS + EXTRA_COLS + list(df.columns[df.columns.str.contains("abs_diff")])

output_files("sensitivity_scenario", cols_to_output, df)


mr = ManagedRun(run_id)
pbi = mr.powerbi.get_db()

chain_run = (
    pbi.db.query("mcmc_run")
    .sort_values(by="loglikelihood", ascending=False)
    .head(1)[["chain", "run"]]
    .values
)
chain, run = tuple(*chain_run)


df = pd.concat(get_mle_outputs(chain, run))
df = df.sort_values(by=BASE_COLS + ["month"])

base_columns = [
    "notifications",
    "hospital_admissions",
    "infection_deaths",
    "hospital_occupancy",
    "non_hosp_notifications",
    "icu_admissions",
    "icu_occupancy",
]

cols_to_output = [
    df_col
    for col in base_columns
    for df_col in df.columns
    if col in df_col and "abs_diff" not in df_col
]
cols_to_output = BASE_COLS + ["month"] + EXTRA_COLS + cols_to_output


output_files("scenario", cols_to_output, df)
