from datetime import datetime
import pandas as pd

from autumn.tools.inputs.database import get_input_db


def weighted_average_google_locations(mob_df, revised_location_map):
    for model_loc, google_locs in revised_location_map.items():
        mob_df[model_loc] = 0
        for g_loc in google_locs:
            mob_df[model_loc] += mob_df[g_loc] * revised_location_map[model_loc][g_loc]
    loc_mobility_values = {loc: mob_df[loc].tolist() for loc in revised_location_map.keys()}
    return loc_mobility_values


def get_mobility_data(country_iso_code: str, region: str, base_date: datetime):
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
    mob_conditions = {"iso3": country_iso_code, "region": region or None}
    mob_df = input_db.query("mobility", conditions=mob_conditions)
    mob_df["date"] = pd.to_datetime(mob_df["date"], format="%Y-%m-%d")
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]
    days = mob_df["date"].apply(lambda d: (d - base_date).days).tolist()
    return mob_df, days
