from datetime import datetime

import pandas as pd

from autumn.inputs.database import get_input_db


def get_mobility_data(country_iso_code: str, region: str, base_date: datetime, location_map: dict):
    """
    Get daily Google mobility data for locations, for a given country.
    Times are in days since a given base date. 

    The location map parameter transforms Google Mobility locations
    into Autumn-friendly locations.

    Google mobility provides us with:
        - workplaces
        - retail_and_recreation
        - grocery_and_pharmacy
        - parks
        - transit_stations
        - residential

    An example mapping would be
    {
        "work": ["workplaces"],
        "other_locations": [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "parks",
            "transit_stations",
        ],
    }
    """
    input_db = get_input_db()
    mob_df = input_db.query(
        "mobility",
        conditions=[
            f"iso3='{country_iso_code}'",
            f"region='{region}'" if region else "region IS NULL",
        ],
    )

    # Average out Google Mobility locations into Autumn-friendly locations
    for new_loc, old_locs in location_map.items():
        mob_df[new_loc] = 0
        for old_loc in old_locs:
            mob_df[new_loc] += mob_df[old_loc]

        mob_df[new_loc] = mob_df[new_loc] / len(old_locs)

    mob_df["date"] = pd.to_datetime(mob_df["date"], format="%Y-%m-%d")
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]
    days = mob_df["date"].apply(lambda d: (d - base_date).days).tolist()
    loc_mobility_values = {loc: mob_df[loc].tolist() for loc in location_map.keys()}
    return loc_mobility_values, days

