from datetime import datetime

import pandas as pd

from autumn.tools.inputs.database import get_input_db


def get_mobility_data(country_iso_code: str, region: str, base_date: datetime):
    """
    Get daily Google mobility data for locations, for a given country.
    Times are in days since a given base date.
    The location map parameter transforms Google Mobility locations into Autumn-friendly locations.
    """

    input_db = get_input_db()
    conditions = {"iso3": country_iso_code, "region": region or None}
    mob_df = input_db.query("mobility", conditions=conditions)
    mob_df["date"] = pd.to_datetime(mob_df["date"], format="%Y-%m-%d")
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]
    days = mob_df["date"].apply(lambda d: (d - base_date).days).tolist()
    return mob_df, days
