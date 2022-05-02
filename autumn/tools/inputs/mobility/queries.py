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

    mov_df = get_movement_data(input_db, country_iso_code, region)
    mob_df = get_google_mobility(country_iso_code, region, base_date, input_db)

    merge_how = "right" if len(mob_df) > 0 else "left"
    mob_df = pd.merge(mov_df, mob_df, on="date", how=merge_how)
    mob_df["iso3"] = country_iso_code
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]

    days = mob_df["date"].apply(lambda d: (d - base_date).days).tolist()
    return mob_df, days


def get_google_mobility(country_iso_code, region, base_date, input_db):
    conditions = {"iso3": country_iso_code, "region": region or None}
    mob_df = input_db.query("mobility", conditions=conditions)
    mob_df["date"] = pd.to_datetime(mob_df["date"], format="%Y-%m-%d")
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]
    return mob_df


def get_movement_data(input_db, country_iso_code: str, region: str):
    """
    Get daily Facebook mobility data for locations, for a given country.
    Times are in days since a given base date.

    """

    if region is None:
        conditions = {"country": country_iso_code}
    else:
        conditions = {"country": country_iso_code, "polygon_name": region}

    mov_df = input_db.query("movement", conditions=conditions)
    mov_df["date"] = pd.to_datetime(mov_df["date"], format="%Y-%m-%d")
    mov_df = mov_df.sort_values(["date"])
    mov_df = mov_df.rename(
        columns={
            "all_day_bing_tiles_visited_relative_change": "tiles_visited",
            "all_day_ratio_single_tile_users": "single_tile",
        }
    )
    # mov_df = mov_df[mov_df["date"] >= base_date]
    mov_df["tiles_visited"] += 1

    if region is None:
        mov_df = mov_df.groupby(["date", "date_index"], as_index=False).mean()

    return mov_df
