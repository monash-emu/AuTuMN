import pandas as pd

from autumn.tools.db import Database

from .fetch import COVID_AU_CSV_PATH, COVID_LGA_CSV_PATH, MOBILITY_LGA_PATH, CLUSTER_MAP
from autumn.tools.utils.utils import create_date_index, COVID_BASE_DATETIME

def preprocess_covid_au(input_db: Database):
    df = pd.read_csv(COVID_AU_CSV_PATH)
    input_db.dump_df("covid_au", df)
    df = pd.read_csv(COVID_LGA_CSV_PATH)
    df = reshape_to_clusters(df)
    input_db.dump_df("covid_dhhs_test", df)

    #df = reshape_vac_to_clusters()
    #input_db.dump_df("covid_dhhs_victoria_2021", df)

def reshape_to_clusters(lga_test):
    """
    Takes the lga testing data frame and creates new DHHS health cluster testing values.

    Input: Pandas data frame of LGA testing
    Output: Pandas data frame of DHHS health clusters testing
    """

    # Read in LGA proportion and removed undesired LGAs.
    lga_df = pd.read_csv(MOBILITY_LGA_PATH)

    # Drop missing dates and LGA's from other states
    lga_test = lga_test[lga_test.CollectionDate.notnull()]
    lga_test.CollectionDate = pd.to_datetime(lga_test.CollectionDate, format="%Y-%m-%d")
    lga_test = lga_test[lga_test.LGA.isin(lga_df.lga_name)]
    lga_test = lga_test[lga_test.CollectionDate >= "2020-01-01"]

    # Calculate LGA and health cluster testing proportions.
    lga_df = pd.merge(lga_test, lga_df, how="left", left_on="LGA", right_on="lga_name")
    lga_df["lga_test_prop"] = lga_df.proportion * lga_df.n
    lga_df = (
        lga_df.groupby(["CollectionDate", "cluster_name"])
        .sum()
        .reset_index()[["CollectionDate", "cluster_name", "lga_test_prop"]]
    )
    lga_df.rename(columns={"CollectionDate": "date", "lga_test_prop": "test"}, inplace=True)

    return lga_df


def  reshape_vac_to_clusters():
    
    vac_df = pd.read_csv(COVID_DHHS_VAC_CSV)
    vac_df["week"] = pd.to_datetime(
        vac_df["week"], format="%d/%m/%Y %H:%M:%S", infer_datetime_format=True
    ).dt.date
    vac_df = create_date_index(COVID_BASE_DATETIME, vac_df, "week")


    cluster_map_df = pd.read_csv(MOBILITY_LGA_PATH)
    vac_df = vac_df.merge(cluster_map_df, left_on="lga_name_2018", right_on="lga_name", how="left")

    vac_df = vac_df[
        [
            "date",
            "date_index",
            "age_group",
            "vaccine_brand_name",
            "dose_1",
            "dose_2",
            "cluster_id",
            "proportion",
        ]
    ]

    # TODO use the postcode to figure out the LGA then map to cluster_id
    vac_df.loc[vac_df.cluster_id.isna(), ["proportion", "cluster_id"]] = [1, 0]
    vac_df.cluster_id.replace(CLUSTER_MAP, inplace=True)
    vac_df.dose_1 = vac_df.dose_1 * vac_df.proportion
    vac_df.dose_2 = vac_df.dose_2 * vac_df.proportion

    vac_df = (
        vac_df.groupby(
            ["date", "date_index", "age_group", "vaccine_brand_name", "cluster_id"]
        )
        .sum()
        .reset_index()
    )

    vac_df[["start_age", "end_age"]] = vac_df.age_group.str.split("-", expand=True)
    vac_df.loc[vac_df.start_age=='85+', ["start_age", "end_age"]] = ['85','89']
    vac_df[["start_age", "end_age"]] = vac_df[["start_age", "end_age"]].apply(pd.to_numeric)

    return vac_df


