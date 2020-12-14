import pandas as pd
import os

from autumn import constants
from requests import get

HOSPITAL_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "hospitalisation_data")
SWEDEN_ICU_PREV = os.path.join(
    HOSPITAL_DIRPATH,
    "Antal som intensivvårdas med Covid-19 per dag - ICU PREVALENCE.xlsx",
)
SWEDEN_ICU_INCID = os.path.join(
    HOSPITAL_DIRPATH, "Antal nyinskrivna vårdtillfällen med Coronavirus - ICU INCIDENCE.xlsx"
)

SPAIN_HOSP = os.path.join(HOSPITAL_DIRPATH, "spain_incid_hosp.csv")

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


def get_france():
    france_incid = pd.read_csv(
        "https://www.data.gouv.fr/en/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c", delimiter=";"
    )

    france_prev = pd.read_csv(
        "https://www.data.gouv.fr/en/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7", delimiter=";"
    )

    france_prev["date_check"] = pd.to_datetime(
        france_prev["jour"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    france_prev.loc[france_prev.date_check.isna(), "date_check"] = pd.to_datetime(
        france_prev["jour"], errors="coerce", format="%d/%m/%Y", infer_datetime_format=False
    )

    france_prev["jour"] = france_prev.date_check

    france_incid["jour"] = pd.to_datetime(
        france_incid["jour"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    france_prev = france_prev.groupby(["jour"]).sum()
    france_incid = france_incid.groupby(["jour"]).sum()

    france_df = france_incid.merge(france_prev, how="outer", on="jour")
    france_df.drop(columns="sexe", inplace=True)
    france_df.index.name = "date"

    france_df.rename(
        columns={
            "incid_hosp": "fra_incid_hosp",
            "incid_rea": "fra_incid_icu",
            "incid_dc": "fra_daily_death",
            "incid_rad": "fra_daily_return_home",
            "hosp": "fra_prev_hosp",
            "rea": "fra_prev_icu",
            "rad": "fra_total_return_home",
            "dc": "fra_total_death_hosp",
        },
        inplace=True,
    )
    return france_df


def get_belgium():
    belgium_df = pd.read_csv("https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv")
    belgium_df["DATE"] = pd.to_datetime(
        belgium_df["DATE"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    belgium_df.drop(columns=["NR_REPORTING"], inplace=True)
    belgium_df.rename(
        columns={
            "TOTAL_IN": "bel_prev_hops",
            "TOTAL_IN_ICU": "bel_prev_icu",
            "TOTAL_IN_RESP": "bel_prev_resp",
            "TOTAL_IN_ECMO": "bel_prev_ecmo",
            "NEW_IN": "bel_incid_hosp_in",
            "NEW_OUT": "bel_incid_hosp_out",
        },
        inplace=True,
    )
    belgium_df = belgium_df.groupby(["DATE"]).sum()
    belgium_df.index.name = "date"

    return belgium_df


def get_italy():
    italy_df = pd.read_csv(
        "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    )

    italy_df.rename(
        columns={
            "data": "date",
            "ricoverati_con_sintomi": "ita_incid_hosp",
            "terapia_intensiva": "ita_incid_icu",
            "totale_ospedalizzati": "ita_prev_hosp",
        },
        inplace=True,
    )
    italy_df = italy_df[["date", "ita_incid_hosp", "ita_incid_icu", "ita_prev_hosp"]]
    italy_df["date"] = pd.to_datetime(
        italy_df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    ).dt.normalize()
    italy_df = italy_df.groupby(["date"]).sum()

    return italy_df


def get_sweden():
    sweden_icu_incid = pd.read_excel(SWEDEN_ICU_INCID, header=1)
    sweden_icu_prev = pd.read_excel(SWEDEN_ICU_PREV, header=1)

    sweden_icu_incid["Datum"] = pd.to_datetime(
        sweden_icu_incid["Datum"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    sweden_icu_prev["Datum"] = pd.to_datetime(
        sweden_icu_prev["Datum"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    sweden_icu_prev.rename(
        columns={"COVID-19 Totalt": "swe_prev_icu", "Datum": "date"}, inplace=True
    )
    sweden_icu_incid.rename(
        columns={"Antal unika personer": "swe_incid_icu", "Datum": "date"}, inplace=True
    )

    sweden_icu_prev = sweden_icu_prev[["date", "swe_prev_icu"]].groupby(["date"]).sum()
    sweden_icu_incid = sweden_icu_incid[["date", "swe_incid_icu"]].groupby(["date"]).sum()

    sweden_df = sweden_icu_incid.merge(sweden_icu_prev, how="outer", on="date")

    return sweden_df


def get_uk():

    uk_df = get_data(endpoint)
    uk_df = pd.DataFrame(uk_df["data"])
    uk_df["date"] = pd.to_datetime(
        uk_df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    uk_df.rename(
        columns={
            "covidOccupiedMVBeds": "uk_prev_icu?",
            "newAdmissions": "uk_incid_hosp",
            "hospitalCases": "uk_prev_hosp",
        },
        inplace=True,
    )
    return uk_df.groupby(["date"]).sum()


def get_spain():

    spain_df = pd.read_csv(SPAIN_HOSP, names=["date", "esp_incid_hosp"])
    spain_df["date"] = pd.to_datetime(
        spain_df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    return spain_df.groupby(["date"]).sum()


def main():
    france_df = get_france()
    belgium_df = get_belgium()
    italy_df = get_italy()
    sweden_df = get_sweden()
    uk_df = get_uk()
    spain_df = get_spain()
    european_data = pd.concat([france_df, belgium_df, italy_df, sweden_df, uk_df, spain_df])
    european_data = european_data.groupby("date").sum()
    european_data.to_csv(os.path.join(HOSPITAL_DIRPATH, "european_data.csv"))


if __name__ == "__main__":
    main()
