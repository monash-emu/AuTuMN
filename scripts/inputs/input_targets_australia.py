import requests
from bs4 import BeautifulSoup
import pandas as pd

CASES_NAMES = ["region", "date", "cases", "7_day_rolling_cases"]
HOSPITAL_ICU_NAMES = [
    "region",
    "date",
    "7_day_average_hospitalised",
    "7_day_average_ICU",
    "7_day_average_cases",
]

# Making a GET request
r = requests.get("https://www.health.gov.au/health-alerts/covid-19/case-numbers-and-statistics/")


# Parsing the HTML
soup = BeautifulSoup(r.content, "html.parser")


def fix_df(df: pd.DataFrame, cols_list: list) -> pd.Dataframe:
    """Tidy up the dataframe column types"""

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["region"] = df["region"].str.upper()
    for col in cols_list:
        df[col] = df[col].replace("", "-99")
        df[col] = df[col].astype("int")
        df.loc[df[col] == -99, col] = pd.NA

    df = df.sort_values(by=["date"])
    df = df.reset_index(drop=True)

    return df


def get_a_dataframe(list_of_data: list, col_names: list) -> pd.DataFrame:
    """A generator that yields a single row dataframe
    for each day and region combination"""

    for case in list_of_data:
        daily_cases_dict = dict(zip(col_names, case.split(";")))
        yield pd.DataFrame(daily_cases_dict, index=[0])


# The HTML tag id for the cases chart
s = soup.find("pre", id="data44742")
cases = s.string.split("!")

df_generator = get_a_dataframe(cases, CASES_NAMES)

df_cases = pd.concat(df_generator)
df_cases = fix_df(df_cases, ["cases", "7_day_rolling_cases"])

# The HTML tag id for the hospitalisation chart
s = soup.find("pre", id="data44764")
hospital_icu = s.string.split("!")

df_generator = get_a_dataframe(hospital_icu, HOSPITAL_ICU_NAMES)

df_hospital_icu = pd.concat(df_generator)
df_hospital_icu = df_hospital_icu.dropna()
cols_to_int = ["7_day_average_hospitalised", "7_day_average_ICU", "7_day_average_cases"]
df_hospital_icu = fix_df(df_hospital_icu, cols_to_int)

# Merge cases and hospitalisation tables
df = df_cases.merge(df_hospital_icu, how="outer", on=["region", "date"])
df
