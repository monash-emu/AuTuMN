
import pandas as pd

from pathlib import Path
from autumn.settings.folders import INPUT_DATA_PATH
from autumn.settings.constants import COVID_BASE_DATETIME

URL = "1kAlNlxhkYv5MF4810gYCTudqiVO5D9xpVNqhzATaFK8"

COVID_BASE_DATETIME = COVID_BASE_DATETIME.date()
INPUT_DATA_PATH =Path(INPUT_DATA_PATH)
VAC_PATH = INPUT_DATA_PATH/ "covid_phl"


file_names = [file for file in list(VAC_PATH.glob('*')) if "NVOC Month END_" in file.stem]
xlsx_file_path = file_names

file_dates = [
    pd.to_datetime(pd.ExcelFile(file).sheet_names[0], format="%m%d%Y").date()
    for file in xlsx_file_path
]

VAC_FILE = dict(zip(xlsx_file_path, file_dates))


def process_df(df, date):

    df = df[:-1]

    col_names = list(df.columns)
    col_names[0] = "vaccination"
    check_unnamed = [True]
    while any(check_unnamed):
        col_names = [
            col_names[col_idx - 1] if "Unnamed" in col_name else col_name
            for col_idx, col_name in enumerate(col_names)
        ]
        check_unnamed = ["unnamed" in col_name.lower() for col_name in col_names]

    col_names = [col.lower().replace(" ", "+") for col in col_names]
    dose_col = list(df.loc[0].str.replace("Sum of CUMULATIVE_", "").str.lower())
    dose_col[0] = ""

    map_col = dict(zip(df.columns, col_names))

    df = df.rename(columns=map_col)

    df = df.T
    df.reset_index(inplace=True)

    df.columns = df[:1].values.tolist()[0]
    df = df[1:]
    df.rename(columns={"Row Labels": "cml_dose"}, inplace=True)
    df["cml_dose"] = df["cml_dose"].str.replace("Sum of CUMULATIVE_", "")
    df["date"] = date

    df["date_index"] = df["date"].apply(lambda d: (d - COVID_BASE_DATETIME).days)
    df["vaccination"] = df["vaccination"].apply(lambda s: s.replace(".1", ""))
    return df


def get_phl_vac(VAC_FILE, process_df):

    dataframe_list = []

    for file, date in VAC_FILE.items():

        df = pd.read_excel(
            file,
            skiprows=1,
            usecols=lambda x: "Total Sum of" not in x,
        )
        df = process_df(df.copy(), date)
        dataframe_list.append(df)
    return pd.concat(dataframe_list)


phl_vac_df = get_phl_vac(VAC_FILE, process_df)
phl_vac_df.to_csv(VAC_PATH / "phl_vaccination.csv", index=False)
