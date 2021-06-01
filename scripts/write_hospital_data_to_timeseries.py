import json
import os

import pandas as pd

from autumn.settings import Region

base_dir = os.path.dirname(os.path.abspath(os.curdir))
HOSP_DATA_FILE = os.path.join(
    base_dir, "data", "inputs", "hospitalisation_data", "european_data.csv"
)

COUNTRY_TARGETS = {
    Region.BELGIUM: ["bel_hosp_adm_per_100K", "new_hospital_admissions"],
    Region.FRANCE: ["fra_hosp_adm_per_100K", "new_hospital_admissions"],
    Region.ITALY: ["ita_hosp_adm_per_100K", "new_hospital_admissions"],
    Region.SPAIN: ["spa_hosp_adm_per_100K", "new_hospital_admissions"],
    Region.SWEDEN: ["swe_icu_adm_per_100K", "new_icu_admissions"],
    Region.UNITED_KINGDOM: ["uk_hosp_adm", "new_hospital_admissions"],
}

POPULATION = {
    Region.BELGIUM: 11589623,
    Region.FRANCE: 65273511,
    Region.ITALY: 60461826,
    Region.SPAIN: 46754778,
    Region.SWEDEN: 10099265,
    Region.UNITED_KINGDOM: 67886011,  # UK not required
}


def convert_year_week_to_day(year_week):
    """
    return a day representing the 4th week defined by year_week
    :param year_week: string such as "2020-W12"
    :param reference_date: date corresponding to time t=0
    :return: integer representing the 4th day of the week
    """
    year = year_week.split("-")[0]
    week = year_week.split("W")[1]

    # work out out date_int, which is the integer representing the first day of the selected week
    ref_integer = -8 if year == "2020" else 363
    date_int = ref_integer + 7 * float(week) + 3

    return date_int


def read_hospital_data(country, data_start_time=0, data_end_time=365):
    path = HOSP_DATA_FILE
    data = pd.read_csv(path)

    mask = pd.notnull(data[COUNTRY_TARGETS[country][0]])
    year_weeks = list(data[mask]["year_week"])
    values = [float(v) for v in data[mask][COUNTRY_TARGETS[country][0]]]

    times = [convert_year_week_to_day(y_w) for y_w in year_weeks]

    min_hosp_time = min(times)
    data_start_time = max(min_hosp_time, data_start_time)
    start_index = next(x[0] for x in enumerate(times) if x[1] >= data_start_time)

    max_hosp_time = max(times)
    data_end_time = min(max_hosp_time, data_end_time)
    end_index = next(x[0] for x in enumerate(times) if x[1] >= data_end_time) - 1

    times = times[start_index : end_index + 1]
    values = values[start_index : end_index + 1]

    return times, values


def write_hospital_data_to_timeseries(country, data_start_time=0, data_end_time=365):
    region = country if country != Region.UNITED_KINGDOM else "united_kingdom"
    target_path = os.path.join(
        "../autumn",
        "projects",
        "covid_19",
        "mixing_optimisation",
        "regions",
        region,
        "timeseries.json",
    )

    times, values = read_hospital_data(country, data_start_time, data_end_time)

    # UK's data represents the average daily number of new hospitalisation for each week.
    # For the other countries, this is total number of new hospitalisations for each week, expressed per 100k people
    if country != Region.UNITED_KINGDOM:
        population = POPULATION[country]
        values = [v / 7 * population / 1.0e5 for v in values]

    with open(target_path, mode="r") as f:
        targets = json.load(f)
        targets[COUNTRY_TARGETS[country][1]]["times"] = times
        targets[COUNTRY_TARGETS[country][1]]["values"] = values
    with open(target_path, "w") as f:
        json.dump(targets, f, indent=2)


if __name__ == "__main__":
    for country in Region.MIXING_OPTI_REGIONS:
        write_hospital_data_to_timeseries(country, data_end_time=365)
