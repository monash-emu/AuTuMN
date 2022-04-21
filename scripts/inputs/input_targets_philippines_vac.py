import os
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from autumn.settings.folders import INPUT_DATA_PATH
from autumn.settings.constants import COVID_BASE_DATETIME
from zipfile import ZipFile

URL = "1kAlNlxhkYv5MF4810gYCTudqiVO5D9xpVNqhzATaFK8"


# zfile = [file for file in os.listdir(VAC_PATH) if 'drive-download' in file]
# zfile = os.path.join(VAC_PATH, zfile[0])

# with ZipFile(zfile) as z:
#     z.extractall(VAC_PATH)

VAC_PATH = os.path.join(INPUT_DATA_PATH, "covid_phl")
file_names = [files for files in os.listdir(VAC_PATH) if ".xlsx" in files]
xlsx_file_path = [os.path.join(VAC_PATH, file) for file in file_names]

file_dates = [pd.to_datetime(date.replace(".xlsx", "")).date() for date in file_names]

VAC_FILE = dict(zip(xlsx_file_path, file_dates))


def process_df(df, date):

    df = df[:-1]
    df = df.replace(
        {
            "Unnamed: 0": {
                "ARMM": "AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)",
                "CAR": "CORDILLERA ADMINISTRA TIVE REGION (CAR)",
                "NCR": "NATIONAL CAPITAL REGION (NCR)",
            }
        }
    )
    col_names = list(df.columns)
    col_names[0] = "vaccination"
    check_unnamed = [True]
    while any(check_unnamed):
        col_names = [
            col_names[col_idx - 1] if "Unnamed" in col_name else col_name
            for col_idx, col_name in enumerate(col_names)
        ]
        check_unnamed = [
            True if "unnamed" in col_name.lower() else False for col_name in col_names
        ]

    col_names = [col.lower().replace(" ", "+") for col in col_names]
    dose_col = list(df.loc[0].str.replace("Sum of CUMULATIVE_", "").str.lower())
    dose_col[0] = ""

    map_col = dict(zip(df.columns, col_names))

    df.rename(columns=map_col, inplace=True)

    df = df.T
    df.reset_index(inplace=True)

    df.columns = df[:1].values.tolist()[0]
    df = df[1:]
    df.rename(columns={"Row Labels": "cml_dose"}, inplace=True)
    df["cml_dose"] = df["cml_dose"].str.replace("Sum of CUMULATIVE_", "")
    df["date"] = date

    df["date_index"] = df["date"].apply(lambda d: (d - COVID_BASE_DATETIME.date()).days)
    df["vaccination"] = df["vaccination"].apply(lambda s: s.replace(".1", ""))
    return df


def get_phl_vac(xlsx_file_path, VAC_FILE, process_df):

    dataframe_list = []

    for file, date in VAC_FILE.items():

        df_dict = pd.read_excel(
            file,
            sheet_name=[0, "Booster"],
            skiprows=1,
            usecols=lambda x: "Total Sum of" not in x,
        )

        df_vac = process_df(df_dict[0].copy(), date)
        df_booster = process_df(df_dict["Booster"].copy(), date)
        dataframe_list.append(df_vac)
        dataframe_list.append(df_booster)

    return pd.concat(dataframe_list)


phl_vac_df = get_phl_vac(xlsx_file_path, VAC_FILE, process_df)
phl_vac_df.to_csv(os.path.join(VAC_PATH, "phl_vaccination.csv"), index=False)


# dataframe_list[5].columns


# col_names = [f"{a}_{b}" if b != "" else a for a, b in zip(col_names, dose_col)]
# df.columns = col_names

# df = df[~df["region"].isin({"Row Labels", "Grand Total"})]


# df.set_index("region", inplace=True)
# df.index

# pd.melt(df, ["region"], value_vars=col_names[1:])


# list(df.loc[0].str.replace("Sum of CUMULATIVE_", "").str.lower())


# gdd.download_file_from_google_drive(URL, VAC_PATH)


# VAC_URL = f"https://docs.google.com/spreadsheets/d/{URL}/export?format=xlsx"
# df = pd.read_excel(VAC_URL)

# f"https://spreadsheets.google.com/feeds/download/spreadsheets/Export?key={URL}&exportFormat=csv&gid=0"


# https://docs.google.com/spreadsheets/d/1kAlNlxhkYv5MF4810gYCTudqiVO5D9xpVNqhzATaFK8/export?format=xlsx
# https://docs.google.com/spreadsheets/d/1Gztm9o8JEPibPWEDwH54qBG5kwj51ILDOk_dxK6uTSY/export?format=csv&gid=1330027783
