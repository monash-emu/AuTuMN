import pandas as pd

from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.db import Database
from autumn.tools.utils.utils import create_date_index

from .fetch import (
    COVID_AU_CSV_PATH,
    COVID_AU_YOUGOV,
    COVID_LGA_CSV_PATH,
    COVID_VAC_COV_CSV,
    COVID_VIDA_POP_CSV,
    COVID_VIDA_VAC_CSV,
    MOBILITY_LGA_PATH,
)


def preprocess_covid_au(input_db: Database):
    df = pd.read_csv(COVID_AU_CSV_PATH)
    input_db.dump_df("covid_au", df)
    df = pd.read_csv(COVID_LGA_CSV_PATH)
    df = reshape_to_clusters(df)
    input_db.dump_df("covid_dhhs_test", df)
    df = pd.read_csv(COVID_VAC_COV_CSV)
    input_db.dump_df("vic_2021", df)  # True vaccination numbers
    df = pd.read_csv(COVID_AU_YOUGOV)
    df = process_yougov(df)
    input_db.dump_df("yougov_vic", df)
    df = pd.read_csv(COVID_VIDA_VAC_CSV)
    input_db.dump_df("vida_vac_model", df)
    df = pd.read_csv(COVID_VIDA_POP_CSV)
    input_db.dump_df("vida_pop", df)


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
    lga_df.rename(
        columns={"CollectionDate": "date", "lga_test_prop": "test"}, inplace=True
    )

    return lga_df


def process_yougov(df):

    df = df[df.state.str.lower() == "victoria"]
    df["endtime"] = pd.to_datetime(df.endtime, format="%d/%m/%Y %H:%M").dt.date
    df = create_date_index(COVID_BASE_DATETIME, df, "endtime")

    return df
