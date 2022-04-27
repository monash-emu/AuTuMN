import pandas as pd

from autumn.tools.db import Database

from .fetch import COVID_BTN_TEST_PATH, COVID_BTN_VAC_PATH
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


def preprocess_covid_btn(input_db: Database):
    df_dict = pd.read_excel(COVID_BTN_TEST_PATH, sheet_name=["Bhutan", "Thimphu"])
    df = preprocess_testing_numbers(df_dict)
    input_db.dump_df("covid_btn_test", df)
    df_dict = pd.read_excel(COVID_BTN_VAC_PATH, sheet_name=["Bhutan", "Thimphu"])
    df = preprocess_vaccination(df_dict)
    input_db.dump_df("covid_btn_vac", df)


def preprocess_testing_numbers(df_dict):
    df_list = []
    for region, df in df_dict.items():
        df = df.rename(columns=lambda x: "date" if x == "to_date" else x)
        df["region"] = region
        df = create_date_index(COVID_BASE_DATETIME, df, "date")
        df = df[
            [
                "date_index",
                "date",
                "region",
                "total_tests",
            ]
        ]
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


df_dict = pd.read_excel(COVID_BTN_VAC_PATH, sheet_name=["Bhutan", "Thimphu"])


def preprocess_vaccination(df_dict):
    df_list = []
    for region, df in df_dict.items():
        df["region"] = region
        df = create_date_index(COVID_BASE_DATETIME, df, "date")
        df["start_age"] = df["age_group"].apply(
            lambda s: 65 if s == "65+" else int(s.split("-")[0])
        )
        df["end_age"] = df["age_group"].apply(
            lambda s: 100 if s == "65+" else int(s.split("-")[1])
        )
        df = df[
            [
                "date_index",
                "date",
                "region",
                "dose_num",
                "start_age",
                "end_age",
                "num",
            ]
        ]
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)
