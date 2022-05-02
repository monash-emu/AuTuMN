import os
import pandas as pd
import sqlite3


db_path = "C:\\Users\Mili\Projects\covid_pbi\output_test"
list_of_files = os.listdir(db_path)

db_files = [os.path.join(db_path, each) for each in list_of_files if ".db" in each]


ID_COLS = ["chain", "run", "scenario", "times"]


def get_table(conn, table):
    query = f"SELECT * FROM {table}"
    return pd.read_sql_query(query, conn)


def chg_col_type(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


for each in db_files:

    conn = sqlite3.connect(each)
    df = get_table(conn, "derived_outputs")

    if len({"stratification", "value"}.intersection(df.columns)) is 0:
        df = df.melt(id_vars=ID_COLS, var_name="stratification", value_name="value")

        df["agegroup"] = df["stratification"].str.extract(r"(?:agegroup_)(\d{1,2})")
        cols = ["chain", "run", "scenario", "times", "agegroup"]

        df = chg_col_type(df, cols)

        df.to_sql("derived_outputs", conn, if_exists="replace", index=False)

    df = get_table(conn, "scenario")
    df = chg_col_type(df, ["scenario", "start_time"])
    df.to_sql("scenario", conn, if_exists="replace", index=False)

    df = get_table(conn, "uncertainty")
    df = chg_col_type(df, ["scenario"])
    df.to_sql("uncertainty", conn, if_exists="replace", index=False)

    df = get_table(conn, "targets")
    df = chg_col_type(df, ["times"])
    df.to_sql("targets", conn, if_exists="replace", index=False)

    df = get_table(conn, "mcmc_run")
    df = chg_col_type(df, ["accept", "chain", "run", "weight"])
    df.to_sql("mcmc_run", conn, if_exists="replace", index=False)

    df = get_table(conn, "mcmc_params")
    df = chg_col_type(df, ["chain", "run"])
    df.to_sql("mcmc_params", conn, if_exists="replace", index=False)

    conn.close()
