import pandas as pd

from autumn.core.inputs.demography.queries import get_iso3_from_country_name
from autumn.core.inputs.database import get_input_db
from autumn.settings.constants import COVID_BASE_DATETIME

EXTRA_UNCERTAINTY_OUTPUTS = {
    "cumulative_incidence": "Cumulative number of infections",
    "transformed_random_process": "Transformed random process",
    "prop_ever_infected": "Proportion ever infected"
}

REQUESTED_UNC_QUANTILES = [.025, .25, .5, .75, .975]


def get_timeseries_data(iso3):
    """
    Create a dictionarie containing country-specific timeseries. This equivalent to loading data from the timeseries json file in
    other projects.
    Args:
        iso3: iso3 code
    Returns:
        timeseries: A dictionary containing the timeseries

    """

    input_db = get_input_db()
    timeseries = {}

    # read new daily deaths from inputs
    data = input_db.query(
        table_name='owid',
        conditions={"iso_code": iso3},
        columns=["date", "new_deaths"]
    )

    # apply moving average
    # data["smoothed_new_deaths"] = data["new_deaths"].rolling(7).mean()[6:]
    # data.dropna(inplace=True)

    # add daily deaths to timeseries dict
    timeseries["infection_deaths"] = {
        "output_key": "infection_deaths",
        "title": "Daily number of deaths",
        "times": (pd.to_datetime(data["date"])- COVID_BASE_DATETIME).dt.days.to_list(),
        "values": data["new_deaths"].to_list(),
        "quantiles": REQUESTED_UNC_QUANTILES
    }

    # read new daily notifications from inputs
    data = input_db.query(
        table_name='owid',
        conditions={"iso_code": iso3},
        columns=["date", "new_cases"]
    )

    # add daily notifications to timeseries dict
    timeseries["infection_deaths"] = {
        "output_key": "notifications",
        "title": "Daily number of confirmed cases",
        "times": (pd.to_datetime(data["date"])- COVID_BASE_DATETIME).dt.days.to_list(),
        "values": data["new_cases"].to_list(),
        "quantiles": REQUESTED_UNC_QUANTILES
    }

    # Repeat same process for cumulated deaths
    data = input_db.query(
        table_name='owid',
        conditions= {"iso_code": iso3},
        columns=["date", "total_deaths"]
    )
    data.dropna(inplace=True)

    timeseries["cumulative_infection_deaths"] = {
        "output_key": "cumulative_infection_deaths",
        "title": "Cumulative number of deaths",
        "times": (pd.to_datetime(data["date"])- COVID_BASE_DATETIME).dt.days.to_list(),
        "values": data["total_deaths"].to_list(),
        "quantiles": REQUESTED_UNC_QUANTILES
    }

    # add extra derived output with no data to request uncertainty
    for output_key, title in EXTRA_UNCERTAINTY_OUTPUTS.items():
        timeseries[output_key] = {
            "output_key": output_key,
            "title": title,
            "times": [],
            "values": [],
            "quantiles": REQUESTED_UNC_QUANTILES
        }
    return timeseries