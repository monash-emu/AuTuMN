import os
import pandas as pd
import sqlite3


db_path = "M:\Documents\@Projects\Covid_consolidate\output_test"
list_of_files = os.listdir(db_path)

db_files = [os.path.join(db_path, each) for each in list_of_files if ".db" in each]


ID_COLS = ["chain", "run", "scenario", "times"]

for each in db_files:

    conn = sqlite3.connect(each)
    query = "SELECT * FROM derived_outputs"
    df = pd.read_sql_query(query, conn)

    if len({"stratification","value"}.intersection(df.columns)) is 0:
        df = df.melt(id_vars=ID_COLS, var_name="stratification", value_name="value")

        df['agegroup'] = df["stratification"].str.extract(r'(?:agegroup_)(\d{1,2})')
        cols = ['chain', 'run', 'scenario', 'times','agegroup']

        for col in cols:
            df[col] = df[col].astype(float)

        df.to_sql("derived_outputs", conn, if_exists="replace", index=False)

    conn.close()

