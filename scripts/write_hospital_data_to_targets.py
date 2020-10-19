import pandas as pd
import os
import datetime
import json

from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import get_weekly_summed_targets

base_dir = os.path.dirname(os.path.abspath(os.curdir))
HOSP_DATA_FILE = os.path.join(base_dir, "data", "inputs", "hospitalisation_data", "european_data.csv")

country_targets = {
    Region.BELGIUM: ['bel_incid_hosp_in', 'new_hospital_admissions'],
    Region.FRANCE: ['fra_incid_hosp', 'new_hospital_admissions'],
    Region.ITALY: ['ita_incid_hosp', 'new_hospital_admissions'],
    Region.SPAIN: ['esp_incid_hosp', 'new_hospital_admissions'],
    Region.SWEDEN: ['swe_incid_icu', 'new_icu_admissions'],
    Region.UNITED_KINGDOM: ['uk_incid_hosp', 'new_hospital_admissions'],
}

start_reporting = {
    Region.FRANCE:  79,
    Region.UNITED_KINGDOM: 83,
}


def read_hospital_data(country, data_start_time=0, data_end_time=365):
    path = HOSP_DATA_FILE
    data = pd.read_csv(path)
    date_ref = datetime.datetime.strptime("2019-12-31", "%Y-%m-%d")

    mask = pd.notnull(data[country_targets[country][0]])
    dates = list(data[mask]['date'])
    values = [float(v) for v in data[mask][country_targets[country][0]]]
    times = [(datetime.datetime.strptime(d, "%d/%m/%Y") - date_ref).days for d in dates]

    min_hosp_time = min(times)
    data_start_time = max(min_hosp_time, data_start_time)
    start_index = times.index(data_start_time)

    max_hosp_time = max(times)
    data_end_time = min(max_hosp_time, data_end_time)
    end_index = times.index(data_end_time)

    times = times[start_index: end_index + 1]
    values = values[start_index: end_index + 1]

    return times, values


def write_hospital_data_to_targets(country, data_start_time=0, data_end_time=365, weekly_average=True):
    region = country if country != "united-kingdom" else "united_kingdom"
    target_path = os.path.join("../apps", "covid_19", "regions", region, "targets.json")

    times, values = read_hospital_data(country, data_start_time, data_end_time)
    if weekly_average:
        times, values = get_weekly_summed_targets(times, values)

    if country in list(start_reporting.keys()):
        min_time = start_reporting[country]
        first_idx_to_keep = next(x[0] for x in enumerate(times) if x[1] >= min_time)
        times = times[first_idx_to_keep:]
        values = values[first_idx_to_keep:]

    with open(target_path, mode="r") as f:
        targets = json.load(f)
        targets[country_targets[country][1]]["times"] = times
        targets[country_targets[country][1]]["values"] = values
    with open(target_path, "w") as f:
        json.dump(targets, f, indent=2)


if __name__ == "__main__":
    for country in Region.MIXING_OPTI_REGIONS:
        write_hospital_data_to_targets(country, data_end_time=275)  # t_max=275 is 1st October 2020
