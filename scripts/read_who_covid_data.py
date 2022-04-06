import datetime
import json
import os

import pandas as pd

from autumn.projects.covid_19.mixing_optimisation.utils import get_weekly_summed_targets
from autumn.settings import Region

from autumn.models.covid_19.constants import COVID_BASE_DATETIME

base_dir = os.path.dirname(os.path.abspath(os.curdir))
WHO_DATA_FILE = os.path.join(base_dir, "data", "who_covid", "WHO-COVID-19-global-data.csv")
who_country_mapping = {"united-kingdom": "The United Kingdom"}


def read_who_data_from_csv(
    variable="confirmed", country="australia", data_start_time=0, data_end_time=365
):
    """
    Read WHO data from csv files
    :param variable: one of "confirmed", "deaths"
    :param country: country
    """
    if country in who_country_mapping:
        country_name = who_country_mapping[country]
    else:
        country_name = country.title()
    path = WHO_DATA_FILE
    data = pd.read_csv(path)
    times = list(data[data.iloc[:, 2] == country_name].iloc[:, 0])

    date_ref = COVID_BASE_DATETIME.date()
    for i, time_string in enumerate(times):
        components = time_string.split("-")
        time_date = datetime.date(int(components[0]), int(components[1]), int(components[2]))
        delta = time_date - date_ref
        times[i] = delta.days

    min_who_time = min(times)
    data_start_time = max(min_who_time, data_start_time)
    start_index = times.index(data_start_time)

    max_who_time = max(times)
    data_end_time = min(max_who_time, data_end_time)
    end_index = times.index(data_end_time)

    times = times[start_index : end_index + 1]
    if variable == "confirmed":
        values = list(data[data.iloc[:, 2] == country_name].iloc[:, 4])[start_index : end_index + 1]
    elif variable == "deaths":
        values = list(data[data.iloc[:, 2] == country_name].iloc[:, 6])[start_index : end_index + 1]

    return times, values


def drop_who_data_to_timeseries(
    country="australia", data_start_time=0, data_end_time=365, weekly_average=True
):
    data = {}
    output_name = {"confirmed": "notifications", "deaths": "infection_deaths"}
    for variable in ["confirmed", "deaths"]:
        times, values = read_who_data_from_csv(variable, country, data_start_time, data_end_time)
        if weekly_average:
            times, values = get_weekly_summed_targets(times, values)
        data[variable] = {"times": times, "values": values}

    region = country if country != "united-kingdom" else "united_kingdom"
    target_path = os.path.join(
        "../autumn/projects",
        "covid_19",
        "mixing_optimisation",
        "regions",
        region,
        "timeseries.json",
    )

    with open(target_path, mode="r") as f:
        targets = json.load(f)
        for variable in ["confirmed", "deaths"]:
            targets[output_name[variable]]["times"] = data[variable]["times"]
            targets[output_name[variable]]["values"] = data[variable]["values"]
    with open(target_path, "w") as f:
        json.dump(targets, f, indent=2)


if __name__ == "__main__":
    for country in Region.MIXING_OPTI_REGIONS:
        drop_who_data_to_timeseries(country, data_end_time=470)  # t_max=275 is 1st October 2020
