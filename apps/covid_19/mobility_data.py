"""
This file imports Google and Apple mobility data into a pandas DF

"""

from datetime import datetime as dt
from datetime import timedelta
import pandas as pd


def prepare_google_mobility(country="Australia"):

    # Assuming model times 1 is  01/01/2020!
    df = download_mobility_data()

    # TODO - change hard coded column indexing!
    npi_columns = df.columns.tolist()[5:]

    df["times"] = (df["date"] - datetime(2020, 1, 1)).dt.days
    df = df[df["country_region"] == country]

    # change from baseline to percentage
    mod_df = df.apply(lambda x: 1 + x / 100 if x.name in npi_columns else x)

    return mod_df


def download_mobility_data():

    # Apple mobility data
    apple_mobility_url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/2007HotfixDev54/v2/en-us/applemobilitytrends-2020-05-08.csv"

    # Google mobility data
    google_mobility_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=722f3143b586a83f"

    google_mobility_df = pd.read_csv(google_mobility_url)
    google_mobility_df["date"] = google_mobility_df["date"].astype("datetime64[D]")

    return google_mobility_df
