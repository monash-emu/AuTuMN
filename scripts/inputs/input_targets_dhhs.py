import os
import pandas as pd
import numpy as np

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index
from autumn.settings import PASSWORD_ENVAR
from getpass import getpass
from autumn.tools.utils import secrets

COVID_AU_DIRPATH = os.path.join(INPUT_DATA_PATH, "covid_au")

CHRIS_CSV = os.path.join(COVID_AU_DIRPATH, "monitoringreport.secret.csv")
COVID_DHHS_DEATH_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_deaths.secret.csv")
COVID_DHHS_CASE_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_cases.secret.csv")
COVID_DHHS_ADMN_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_admissions.secret.csv")
COVID_DHHS_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_vaccination.secret.csv")
COVID_VIDA_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "vida_vac.secret.csv")
COVID_VIDA_POP_CSV = os.path.join(COVID_AU_DIRPATH, "vida_pop.csv")

COVID_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "vac_cov.csv")

COVID_DHHS_POSTCODE_LGA_CSV = os.path.join(COVID_AU_DIRPATH, "postcode lphu concordance.csv")

COVID_VICTORIA_TARGETS_CSV = os.path.join(
    PROJECTS_PATH, "covid_19", "victoria", "victoria", "targets.secret.json"
)

# Two different mappings
LGA_TO_CLUSTER = os.path.join(
    INPUT_DATA_PATH, "mobility", "LGA to Cluster mapping dictionary with proportions.csv"
)

LGA_TO_HSP = os.path.join(INPUT_DATA_PATH, "covid_au", "LGA_HSP map_v2.csv")

COVID_DHHS_MAPING = LGA_TO_HSP  # This is the new mapping

TODAY = (pd.to_datetime("today") - COVID_BASE_DATETIME).days

TARGET_MAP_DHHS = {
    "notifications": "cluster_cases",
    "hospital_occupancy": "value_hosp",
    "icu_occupancy": "value_icu",
    "icu_admissions": "admittedtoicu",
    "hospital_admissions": "nadmissions",
    "infection_deaths": "cluster_deaths",
}

cluster_map_df = pd.read_csv(COVID_DHHS_MAPING)

map_id = cluster_map_df[["cluster_id", "cluster_name"]].drop_duplicates()
map_id["cluster_name"] = (
    map_id["cluster_name"]
    .str.upper()
    .str.replace("&", "")
    .str.replace("  ", "_")
    .str.replace(" ", "_")
)

CLUSTER_MAP = dict(map_id.values)
CLUSTER_MAP[0] = "VICTORIA"

CHRIS_MAP = {
    # North east metro
    "St Vincents Hospital": "NORTH_EAST_METRO",
    "St Vincents Private Hospital Fitzroy": "NORTH_EAST_METRO",
    "Austin Hospital": "NORTH_EAST_METRO",
    "Northern Hospital, The [Epping]": "NORTH_EAST_METRO",
    "Warringal Private Hospital [Heidelberg]": "NORTH_EAST_METRO",
    "Maroondah Hospital [East Ringwood]": "NORTH_EAST_METRO",
    "Box Hill Hospital": "NORTH_EAST_METRO",
    "Angliss Hospital": "NORTH_EAST_METRO",
    "Epworth Eastern Hospital": "NORTH_EAST_METRO",
    "Knox Private Hospital [Wantirna]": "NORTH_EAST_METRO",
    # South east metro
    "Bays Hospital, The [Mornington]": "SOUTH_EAST_METRO",
    "Frankston Hospital": "SOUTH_EAST_METRO",
    "Peninsula Private Hospital [Frankston]": "SOUTH_EAST_METRO",
    "Holmesglen Private Hospital ": "SOUTH_EAST_METRO",
    "Alfred, The [Prahran]": "SOUTH_EAST_METRO",
    "Cabrini Malvern": "SOUTH_EAST_METRO",
    "Monash Medical Centre [Clayton]": "SOUTH_EAST_METRO",
    "Valley Private Hospital, The [Mulgrave]": "SOUTH_EAST_METRO",
    "Monash Children's Hospital": "SOUTH_EAST_METRO",
    "Dandenong Campus": "SOUTH_EAST_METRO",
    "St John of God Berwick Hospital": "SOUTH_EAST_METRO",
    "Casey Hospital": "SOUTH_EAST_METRO",
    # West metro
    "Royal Childrens Hospital [Parkville]": "WEST_METRO",
    "Sunshine Hospital": "WEST_METRO",
    "Epworth Freemasons": "WEST_METRO",
    "Western Hospital [Footscray]": "WEST_METRO",
    "Melbourne Private Hospital, The [Parkville]": "WEST_METRO",
    "Royal Melbourne Hospital - City Campus": "WEST_METRO",
    "Mercy Public Hospitals Inc [Werribee]": "WEST_METRO",
    "Epworth Hospital [Richmond]": "WEST_METRO",
    "John Fawkner - Moreland Private Hospital": "WEST_METRO",
    # Grampians
    "Ballarat Health Services [Base Campus]": "GRAMPIANS",
    "St John of God Ballarat Hospital": "GRAMPIANS",
    "Wimmera Base Hospital [Horsham]": "GRAMPIANS",
    # Loddon malle
    "Bendigo Hospital, The": "LODDON_MALLEE",
    "New Mildura Base Hospital": "LODDON_MALLEE",
    "St John of God Bendigo Hospital": "LODDON_MALLEE",
    "Mildura Base Public Hospital": "LODDON_MALLEE",
    # Barwon south west
    "St John of God Geelong Hospital": "BARWON_SOUTH_WEST",
    "Geelong Hospital": "BARWON_SOUTH_WEST",
    "South West Healthcare [Warrnambool]": "BARWON_SOUTH_WEST",
    "Epworth Geelong": "BARWON_SOUTH_WEST",
    "Hamilton Base Hospital": "BARWON_SOUTH_WEST",
    # Hume
    "Albury Wodonga Health - Albury": "HUME",
    "Goulburn Valley Health [Shepparton]": "HUME",
    "Northeast Health Wangaratta": "HUME",
    # Gippsland
    "Latrobe Regional Hospital [Traralgon]": "GIPPSLAND",
    "Central Gippsland Health Service [Sale]": "GIPPSLAND",
}

CHRIS_HOSPITAL = "Confirmed COVID ‘+’ cases admitted to your hospital"
CHRIS_ICU = "Confirmed COVID ‘+’ cases in your ICU/HDU(s)"

fix_lga = {
    "Unknown": 0,
    "Kingston (C) (Vic.)": "Kingston (C)",
    "Interstate": 0,
    "Overseas": 0,
    "Melton (C)": "Melton (S)",
    "Latrobe (C) (Vic.)": "Latrobe (C)",
    "Wodonga (C)": "Wodonga (RC)",
    "Unincorporated Vic": 0,
}


def main():

    process_zip_files()
    cases = preprocess_cases()
    cases = load_cases(cases)

    chris_icu = load_chris_df(CHRIS_ICU)
    chris_hosp = load_chris_df(CHRIS_HOSPITAL)

    chris_df = chris_hosp.merge(
        chris_icu, on=["date_index", "cluster_id"], how="outer", suffixes=("_hosp", "_icu")
    )
    chris_df = chris_df.groupby(["date_index", "cluster_id"]).sum().reset_index()

    admissions = preprocess_admissions()
    admissions = load_admissions(admissions)

    deaths = preprocess_deaths()
    deaths = load_deaths(deaths)

    cases = cases.merge(chris_df, on=["date_index", "cluster_id"], how="outer")
    cases = cases.merge(admissions, on=["date_index", "cluster_id"], how="outer")
    cases = cases.merge(deaths, on=["date_index", "cluster_id"], how="outer")

    cases = cases[cases["date_index"] < TODAY]

    password = os.environ.get(PASSWORD_ENVAR, "")
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    for cluster in CLUSTER_MAP.values():
        if cluster == "VICTORIA":
            continue

        cluster_secrets_file = os.path.join(
            PROJECTS_PATH, "covid_19", "victoria", cluster.lower(), "targets.secret.json"
        )

        cluster_df = cases.loc[cases.cluster_id == cluster]

        update_timeseries(TARGET_MAP_DHHS, cluster_df, cluster_secrets_file, password)

    vic_df = cases.groupby("date_index").sum(skipna=True).reset_index()

    update_timeseries(TARGET_MAP_DHHS, vic_df, COVID_VICTORIA_TARGETS_CSV, password)

    # True vaccination numbers
    df = preprocess_vac()
    df = create_vac_coverage(df)

    df.to_csv(COVID_VAC_CSV, index=False)

    # Vida's vaccination model
    df = fetch_vac_model()
    update_vida_pop(df)
    df = preprocess_vac_model(df)

    df.to_csv(COVID_VIDA_VAC_CSV, index=False)
    secrets.write(COVID_VIDA_VAC_CSV, password)


def merge_with_mapping_df(df, left_col_name):

    df = df.merge(cluster_map_df, left_on=[left_col_name], right_on=["lga_name"], how="left")
    df.loc[df.cluster_id.isna(), ["cluster_id", "cluster_name", "proportion"]] = [0, "VIC", 1]
    df.cluster_id.replace(CLUSTER_MAP, inplace=True)

    return df


def preprocess_csv(csv_file, col_name):
    df = pd.read_csv(csv_file)
    df = create_date_index(COVID_BASE_DATETIME, df, col_name)
    return df


def preprocess_admissions():

    df = preprocess_csv(COVID_DHHS_ADMN_CSV, "AdmissionDate")
    df.lga.replace(fix_lga, inplace=True)

    return df


def load_admissions(df):

    df = merge_with_mapping_df(df, "lga")
    df[["admittedtoicu", "ventilated", "nadmissions"]] = df[
        ["admittedtoicu", "ventilated", "nadmissions"]
    ].multiply(df["proportion"], axis="index")
    df = (
        df[["cluster_id", "date_index", "nadmissions", "admittedtoicu", "ventilated"]]
        .groupby(["cluster_id", "date_index"])
        .sum()
        .reset_index()
    )

    return df


def preprocess_cases():

    df = preprocess_csv(COVID_DHHS_CASE_CSV, "DiagnosisDate")
    df = df.groupby(["date_index", "lga"]).sum().reset_index()
    df.lga.replace(fix_lga, inplace=True)

    return df


def load_cases(df):

    df = merge_with_mapping_df(df, "lga")
    df["cluster_cases"] = df.nnewcases * df.proportion
    df = (
        df[["date_index", "cluster_id", "cluster_cases"]]
        .groupby(["date_index", "cluster_id"])
        .sum()
        .reset_index()
    )
    return df


def load_chris_df(load: str):
    """
    Load data from CSV downloaded from CHRIS website
    """
    df = pd.read_csv(CHRIS_CSV)
    df.rename(
        columns={
            "CampusName": "cluster_id",
            "Jurisdiction": "state",
            "FieldName": "type",
            "Value": "value",
            "EffectiveFrom": "E_F",
            "EffectiveTo": "E_T",
        },
        inplace=True,
    )

    df = df[df.type == load][["cluster_id", "state", "value", "E_F"]]
    df["E_F"] = pd.to_datetime(df["E_F"], format="%d/%m/%Y %H:%M:%S", infer_datetime_format=True)
    df = create_date_index(COVID_BASE_DATETIME, df, "E_F")

    df = df.astype({"value": int})
    df = df[["cluster_id", "date_index", "value"]]

    # Sort and remove duplicates to obtain max for a given date.
    df.sort_values(
        by=["cluster_id", "date_index", "value"], ascending=[True, True, False], inplace=True
    )
    df.drop_duplicates(["cluster_id", "date_index"], keep="first", inplace=True)
    df["cluster_id"] = df.cluster_id.replace(CHRIS_MAP)  # .str.lower()

    df = df.groupby(["date_index", "cluster_id"]).sum().reset_index()

    return df


def preprocess_vac():
    "Convert DHHS true vaccinations numbers to LGA"

    df = preprocess_csv(COVID_DHHS_VAC_CSV, "EncounterDate")

    postcode_lga = (
        pd.read_csv(COVID_DHHS_POSTCODE_LGA_CSV, usecols=[0, 1])
        .groupby(["postcode", "lga_name_2018"])
        .size()
        .reset_index()
    )

    df = df.merge(postcode_lga, on="postcode", how="left")
    df.lga_name_2018.replace(fix_lga, inplace=True)

    return df


def create_vac_coverage(df):
    "Creates an aggregated vaccination coveragecsv for inputs db vic_2021(true vaccination numbers)"

    df = merge_with_mapping_df(df, "lga_name_2018")

    df["n"] = df.n * df.proportion

    df = (
        df[["date", "date_index", "agegroup", "dosenumber", "cluster_id", "n"]]
        .groupby(["date", "date_index", "agegroup", "dosenumber", "cluster_id"])
        .sum()
        .reset_index()
    )

    df.sort_values(by=["cluster_id", "agegroup", "date_index"], inplace=True)

    df = create_age_cols(df, {"90-94": "85-89", "95-99": "85-89", "100+": "85-89"})

    numeric_cols = ["dosenumber", "n", "start_age", "end_age"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    return df


def create_age_cols(df, age_map):
    df.agegroup.replace(age_map, inplace=True)
    df["start_age"] = df["agegroup"].apply(lambda s: int(s.split("-")[0]))
    df["end_age"] = df["agegroup"].apply(lambda s: int(s.split("-")[1]))

    return df


def process_zip_files():

    files_map = {
        "nAdmissions_by": COVID_DHHS_ADMN_CSV,
        "_nEncounters_by": COVID_DHHS_VAC_CSV,
        "_NewCases_by_": COVID_DHHS_CASE_CSV,
        "_deaths_LGA": COVID_DHHS_DEATH_CSV,
        "monitoringreport.csv": CHRIS_CSV,
    }

    for file, value in files_map.items():
        for each in os.listdir(COVID_AU_DIRPATH):
            if file in each:
                pd.read_csv(os.path.join(COVID_AU_DIRPATH, each)).to_csv(value, index=False)
                os.remove(os.path.join(COVID_AU_DIRPATH, each))


def preprocess_deaths():

    df = pd.read_csv(COVID_DHHS_DEATH_CSV)
    df = df[~df.DateOfDeath.isna()]

    df = create_date_index(COVID_BASE_DATETIME, df, "DateofDeath")
    df = df.groupby(["date_index", "lga"]).sum().reset_index()
    df.lga.replace(fix_lga, inplace=True)

    return df


def load_deaths(df):

    df = merge_with_mapping_df(df, "lga")
    df["cluster_deaths"] = df.n * df.proportion
    df = (
        df[["date_index", "cluster_id", "cluster_deaths"]]
        .groupby(["date_index", "cluster_id"])
        .sum()
        .reset_index()
    )

    return df


def fetch_vac_model():

    df = pd.read_csv(
        os.path.join(COVID_AU_DIRPATH, "vac_by_week_lga.secret.csv"),
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
            8,
        ],
    )
    create_date_index(COVID_BASE_DATETIME, df, "week")
    df.lga.replace(fix_lga, inplace=True)

    return df


def create_vic_total(df):
    df = df.groupby(["date", "age_group", "vaccine_brand_name", "date_index"], as_index=False).sum()
    df["lga"] = "VICTORIA"

    return df


def preprocess_vac_model(df):

    vic_df = create_vic_total(df)
    df = df.append(vic_df)

    df = merge_with_mapping_df(df, "lga")

    df["dose_1"] = df.dose_1 * df.proportion
    df["dose_2"] = df.dose_2 * df.proportion
    df = df[
        ["vaccine_brand_name", "cluster_id", "age_group", "date_index", "date", "dose_1", "dose_2"]
    ]

    df = df.groupby(
        ["vaccine_brand_name", "cluster_id", "age_group", "date_index", "date"], as_index=False
    ).sum()
    df.sort_values(by=["vaccine_brand_name", "cluster_id", "age_group", "date_index"], inplace=True)
    df.vaccine_brand_name.replace(
        {"COVID-19 Vaccine AstraZeneca": "astra_zeneca", "Pfizer Comirnaty": "pfizer"}, inplace=True
    )
    df.age_group.replace({"85+": "85-89"}, inplace=True)
    df["start_age"] = df["age_group"].apply(lambda s: int(s.split("-")[0]))
    df["end_age"] = df["age_group"].apply(lambda s: int(s.split("-")[1]))

    numeric_cols = ["dose_1", "dose_2", "start_age", "end_age"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    return df


def update_vida_pop(df):

    vic_df = create_vic_total(df)
    df = df.append(vic_df)

    df = df[["lga", "age_group", "popn"]].drop_duplicates()
    df = merge_with_mapping_df(df, "lga")

    df["popn"] = df["popn"] * df["proportion"]
    df = (
        df[["cluster_id", "age_group", "popn"]]
        .groupby(["cluster_id", "age_group"], as_index=False)
        .sum()
    )
    df.age_group.replace({"85+": "85-89"}, inplace=True)
    df["start_age"] = df["age_group"].apply(lambda s: int(s.split("-")[0]))
    df["end_age"] = df["age_group"].apply(lambda s: int(s.split("-")[1]))

    df.to_csv(COVID_VIDA_POP_CSV, index=False)


if __name__ == "__main__":
    main()
