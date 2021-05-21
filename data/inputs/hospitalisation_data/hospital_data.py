import os

import pandas as pd
from requests import get

from settings import INPUT_DATA_PATH

HOSPITAL_DIRPATH = os.path.join(INPUT_DATA_PATH, "hospitalisation_data")


URL = "https://opendata.ecdc.europa.eu/covid19/hospitalicuadmissionrates/csv/data.csv"
COUNTRIES = {"france", "belgium", "italy", "sweden", "uk", "spain"}
RENAME_INDICATOR = {
    "Daily hospital occupancy": "hosp_occup",
    "Daily ICU occupancy": "icu_occup",
    "Weekly new hospital admissions per 100k": "hosp_adm_per_100K",
    "Weekly new ICU admissions per 100k": "icu_adm_per_100K",
}

endpoint = (
    "https://api.coronavirus.data.gov.uk/v1/data?"
    "filters=areaType=overview&"
    'structure={"date":"date","covidOccupiedMVBeds":"covidOccupiedMVBeds","newAdmissions":"newAdmissions","hospitalCases":"hospitalCases"}'
)


def get_data(url):

    response = get(endpoint, timeout=10)

    if response.status_code >= 400:
        raise RuntimeError(f"Request failed: { response.text }")

    return response.json()


def get_uk():

    uk_df = get_data(endpoint)
    uk_df = pd.DataFrame(uk_df["data"])
    uk_df["date"] = pd.to_datetime(
        uk_df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    uk_df.rename(
        columns={
            "covidOccupiedMVBeds": "uk_icu_occup",
            "newAdmissions": "uk_hosp_adm",
            "hospitalCases": "uk_hosp_occup",
        },
        inplace=True,
    )

    uk_df["year_week"] = uk_df.date.dt.strftime(date_format="%Y-W%U")

    # Need to renumber week 0 to last week of 2020
    uk_df.year_week.replace({"2021-W00": "2020-W53"}, inplace=True)

    uk_df.groupby(["year_week"]).mean().reset_index()

    return uk_df.groupby(["year_week"]).mean().reset_index()


def get_eu_countries():
    eu_countries = pd.read_csv(URL)
    eu_countries.country = eu_countries.country.str.lower()
    eu_countries = eu_countries.loc[
        eu_countries.country.isin(COUNTRIES), ["country", "indicator", "year_week", "value"]
    ]
    eu_countries.indicator.replace(RENAME_INDICATOR, inplace=True)

    eu_countries_daily = eu_countries.loc[eu_countries.indicator.isin(["hosp_occup", "icu_occup"])]
    eu_countries = eu_countries.loc[~eu_countries.indicator.isin(["hosp_occup", "icu_occup"])]
    eu_countries_daily = eu_countries_daily.groupby(["country", "indicator", "year_week"]).mean()
    eu_countries_daily.reset_index(inplace=True)

    eu_countries = eu_countries.append(eu_countries_daily)
    eu_countries["cntry_ind"] = eu_countries.country.str[:3] + "_" + eu_countries.indicator

    eu_countries = eu_countries.pivot_table(
        index=["year_week"], columns="cntry_ind", values="value"
    ).reset_index()

    return eu_countries


def main():

    uk_df = get_uk()
    eu_countries = get_eu_countries()
    european_data = eu_countries.merge(uk_df, how="outer", on="year_week")
    european_data.to_csv(os.path.join(HOSPITAL_DIRPATH, "european_data.csv"))


if __name__ == "__main__":
    main()
