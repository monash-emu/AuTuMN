import os
import pandas as pd

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.tools.utils.utils import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index
from autumn.settings import PASSWORD_ENVAR
from getpass import getpass


# Use OWID csv for notification and death numbers.
COVID_AU_DIRPATH = os.path.join(INPUT_DATA_PATH, "covid_au")
COVID_AU_CSV_PATH = os.path.join(COVID_AU_DIRPATH, "COVID_AU_state_daily_change.csv")

CHRIS_CSV = os.path.join(COVID_AU_DIRPATH, "monitoringreport.secret.csv")

COVID_DHHS_CASE_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_cases.secret.csv")
COVID_DHHS_ADMN_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_admissions.secret.csv")
COVID_DHHS_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "monashmodelextract_vaccination.secret.csv")
COVID_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "vac_cov.csv")


COVID_DHHS_POSTCODE_LGA_CSV = os.path.join(COVID_AU_DIRPATH, "postcode lphu concordance.csv")

COVID_VIC2021_TARGETS_CSV = os.path.join(
    PROJECTS_PATH, "covid_19", "victoria", "victoria_2021", "targets.secret.json"
)
COVID_DHHS_CLUSTERS_CSV = os.path.join(
    INPUT_DATA_PATH, "mobility", "LGA to Cluster mapping dictionary with proportions.csv"
)

CLUSTER_MAP = {
    1: "NORTH_METRO",
    2: "SOUTH_EAST_METRO",
    3: "SOUTH_METRO",
    4: "WEST_METRO",
    5: "BARWON_SOUTH_WEST",
    6: "GIPPSLAND",
    7: "GRAMPIANS",
    8: "HUME",
    9: "LODDON_MALLEE",
    0: "VIC",
}

CHRIS_MAP = {
    "Royal Childrens Hospital [Parkville]": "WEST_METRO",
    "Alfred, The [Prahran]": "SOUTH_METRO",
    "Cabrini Malvern": "SOUTH_METRO",
    "Ballarat Health Services [Base Campus]": "GRAMPIANS",
    "Albury Wodonga Health - Albury": "HUME",
    "Epworth Freemasons": "WEST_METRO",
    "Sunshine Hospital": "WEST_METRO",
    "Western Hospital [Footscray]": "WEST_METRO",
    "St Vincents Hospital": "NORTH_METRO",
    "Bendigo Hospital, The": "LODDON_MALLEE",
    "Bays Hospital, The [Mornington]": "SOUTH_METRO",
    "Latrobe Regional Hospital [Traralgon]": "GIPPSLAND",
    "Peninsula Private Hospital [Frankston]": "SOUTH_METRO",
    "Royal Melbourne Hospital - City Campus": "WEST_METRO",
    "Melbourne Private Hospital, The [Parkville]": "WEST_METRO",
    "St John of God Geelong Hospital": "BARWON_SOUTH_WEST",
    "Maroondah Hospital [East Ringwood]": "SOUTH_EAST_METRO",
    "Frankston Hospital": "SOUTH_METRO",
    "St Vincents Private Hospital Fitzroy": "NORTH_METRO",
    "New Mildura Base Hospital": "LODDON_MALLEE",
    "Box Hill Hospital": "SOUTH_EAST_METRO",
    "Austin Hospital": "NORTH_METRO",
    "Angliss Hospital": "SOUTH_EAST_METRO",
    "Geelong Hospital": "BARWON_SOUTH_WEST",
    "Monash Medical Centre [Clayton]": "SOUTH_EAST_METRO",
    "Goulburn Valley Health [Shepparton]": "HUME",
    "Warringal Private Hospital [Heidelberg]": "NORTH_METRO",
    "St John of God Ballarat Hospital": "GRAMPIANS",
    "Epworth Eastern Hospital": "SOUTH_EAST_METRO",
    "South West Healthcare [Warrnambool]": "BARWON_SOUTH_WEST",
    "Northeast Health Wangaratta": "HUME",
    "Mercy Public Hospitals Inc [Werribee]": "WEST_METRO",
    "Epworth Hospital [Richmond]": "WEST_METRO",
    "Holmesglen Private Hospital ": "SOUTH_METRO",
    "Knox Private Hospital [Wantirna]": "SOUTH_EAST_METRO",
    "St John of God Bendigo Hospital": "LODDON_MALLEE",
    "Wimmera Base Hospital [Horsham]": "GRAMPIANS",
    "Valley Private Hospital, The [Mulgrave]": "SOUTH_EAST_METRO",
    "John Fawkner - Moreland Private Hospital": "WEST_METRO",
    "Epworth Geelong": "BARWON_SOUTH_WEST",
    "Monash Children's Hospital": "SOUTH_EAST_METRO",
    "Central Gippsland Health Service [Sale]": "GIPPSLAND",
    "Northern Hospital, The [Epping]": "NORTH_METRO",
    "Dandenong Campus": "SOUTH_EAST_METRO",
    "Hamilton Base Hospital": "BARWON_SOUTH_WEST",
    "St John of God Berwick Hospital": "SOUTH_EAST_METRO",
    "Casey Hospital": "SOUTH_EAST_METRO",
    "Mildura Base Public Hospital": "LODDON_MALLEE",
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

    cases = cases.merge(chris_df, on=["date_index", "cluster_id"], how="outer")
    cases = cases.merge(admissions, on=["date_index", "cluster_id"], how="outer")

    password = os.environ.get(PASSWORD_ENVAR, "")
    if not password:
        password = getpass(prompt="Enter the encryption password:")

    for cluster in CLUSTER_MAP.values():

        TARGET_MAP_DHHS = {
            f"notifications_for_cluster_{cluster.lower()}": "cluster_cases",
            f"hospital_occupancy_for_cluster_{cluster.lower()}": "value_hosp",
            f"icu_occupancy_for_cluster_{cluster.lower()}": "value_icu",
            f"icu_admissions_for_cluster_{cluster.lower()}": "admittedtoicu",
            f"hospital_admissions_for_cluster_{cluster.lower()}": "nadmissions",
        }

        cluster_df = cases.loc[cases.cluster_id == cluster]

        update_timeseries(TARGET_MAP_DHHS, cluster_df, COVID_VIC2021_TARGETS_CSV, password)

    vic_df = cases.groupby("date_index").sum().reset_index()

    TARGET_MAP_DHHS = {
        "notifications": "cluster_cases",
        "hospital_occupancy": "value_hosp",
        "icu_occupancy": "value_icu",
        "icu_admissions": "admittedtoicu",
        "hospital_admissions": "nadmissions",
    }

    update_timeseries(TARGET_MAP_DHHS, vic_df, COVID_VIC2021_TARGETS_CSV, password)

    df = preprocess_vac()
    df = create_vac_coverage(df)

    df.to_csv(COVID_VAC_CSV)


def preprocess_admissions():
    df = pd.read_csv(COVID_DHHS_ADMN_CSV)
    df = create_date_index(COVID_BASE_DATETIME, df, "AdmissionDate")

    df.lga.replace(fix_lga, inplace=True)
    return df


def load_admissions(df):

    cluster_map_df = pd.read_csv(COVID_DHHS_CLUSTERS_CSV)
    df = df.merge(cluster_map_df, left_on=["lga"], right_on=["lga_name"], how="left")
    df.loc[df.cluster_id.isna(), ["cluster_id", "cluster_name", "proportion"]] = [0, "VIC", 1]
    df.cluster_id.replace(CLUSTER_MAP, inplace=True)

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
    df = pd.read_csv(COVID_DHHS_CASE_CSV)
    df = create_date_index(COVID_BASE_DATETIME, df, "DiagnosisDate")
    df = df.groupby(["date_index", "lga"]).sum().reset_index()
    df.lga.replace(fix_lga, inplace=True)

    return df


def load_cases(df):
    cluster_map_df = pd.read_csv(COVID_DHHS_CLUSTERS_CSV)

    df = df.merge(cluster_map_df, left_on=["lga"], right_on=["lga_name"], how="left")

    df.loc[df.cluster_id.isna(), ["cluster_id", "cluster_name", "proportion"]] = [0, "VIC", 1]

    df["cluster_cases"] = df.nnewcases * df.proportion
    df = (
        df[["date_index", "cluster_id", "cluster_cases"]]
        .groupby(["date_index", "cluster_id"])
        .sum()
        .reset_index()
    )
    df.cluster_id.replace(CLUSTER_MAP, inplace=True)
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

    df = pd.read_csv(COVID_DHHS_VAC_CSV)
    postcode_lga = (
        pd.read_csv(COVID_DHHS_POSTCODE_LGA_CSV, usecols=[0, 1])
        .groupby(["postcode", "lga_name_2018"])
        .size()
        .reset_index()
    )

    df = create_date_index(COVID_BASE_DATETIME, df, "EncounterDate")

    df = df.merge(postcode_lga, on="postcode", how="left")
    df.lga_name_2018.replace(fix_lga, inplace=True)

    return df


def create_vac_coverage(df):

    cluster_map_df = pd.read_csv(COVID_DHHS_CLUSTERS_CSV)
    cluster_map_df["cluster_pop"] = cluster_map_df.proportion * cluster_map_df.population
    cluster_map_df.cluster_id.replace(CLUSTER_MAP, inplace=True)
    cluster_pop = (
        cluster_map_df.groupby("cluster_id").sum().reset_index()[["cluster_id", "cluster_pop"]]
    )

    df = df.merge(cluster_map_df, left_on=["lga_name_2018"], right_on=["lga_name"], how="left")
    df.loc[df.cluster_id.isna(), ["cluster_id", "cluster_name", "proportion"]] = [0, "VIC", 1]
    df.cluster_id.replace(CLUSTER_MAP, inplace=True)

    df["n"] = df.n * df.proportion

    df = (
        df[["date", "date_index", "agegroup", "dosenumber", "cluster_id", "n"]]
        .groupby(["date", "date_index", "agegroup", "dosenumber", "cluster_id"])
        .sum()
        .reset_index()
    )

    df = df.merge(cluster_pop, on=["cluster_id"], how="left")
    df["coverage"] = df.n / df.cluster_pop
    df.sort_values(by=["cluster_id", "agegroup", "date_index"], inplace=True)

    return df


    
if __name__ == "__main__":
    main()

