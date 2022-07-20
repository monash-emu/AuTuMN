from typing import List

import pandas as pd

GITHUB_MOH = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/"

MOH_FILES = [
    "cases_malaysia",
    "deaths_malaysia",
    "hospital",
    "icu",
    "cases_state",
    "deaths_state",
]


def fetch_mys_data(base_url: str = GITHUB_MOH, file_list: List[str] = MOH_FILES) -> pd.DataFrame:
    """
    Request files from MoH and combine them into one data frame.
    """
    a_list = []
    for file in file_list:
        data_type = file.split("_")[0]
        df = pd.read_csv(base_url + file + ".csv")
        df["type"] = data_type
        a_list.append(df)
    df = pd.concat(a_list)

    df.loc[df["state"].isna(), "state"] = "Malaysia"
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def get_target_observations(df: pd.DataFrame, region: str, obs_type: str):
    mask = (df["state"] == region) & (df["type"] == obs_type)
    return df[mask]


def get_initial_population(region: str):
    # Download population data
    population_url = (
        "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv"
    )
    df_pop = pd.read_csv(population_url)
    initial_population = df_pop[df_pop["state"] == region]["pop"][0]
    return initial_population
