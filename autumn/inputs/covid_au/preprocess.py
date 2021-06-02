import pandas as pd

from autumn.db import Database

from .fetch import COVID_AU_CSV_PATH, COVID_LGA_CSV_PATH, MOBILITY_LGA_PATH


def preprocess_covid_au(input_db: Database):
    df = pd.read_csv(COVID_AU_CSV_PATH)
    input_db.dump_df("covid_au", df)
    df = pd.read_csv(COVID_LGA_CSV_PATH)
    df = reshape_to_clusters(df)
    input_db.dump_df("covid_dhhs_test", df)


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
