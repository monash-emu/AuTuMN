import pandas as pd
import os
import datetime

WHO_DATA_FILE = os.path.join("../data", "who_covid", f"WHO-COVID-19-global-data.csv")
country_mapping = {"united-kingdom": "The United Kingdom"}


def read_who_data_from_csv(
    variable="confirmed", country="Australia", data_start_time=0, data_end_time=365
):
    """
    Read WHO data from csv files
    :param variable: one of "confirmed", "deaths"
    :param country: country
    """
    if country in country_mapping:
        country_name = country_mapping[country]
    else:
        country_name = country.title()
    path = WHO_DATA_FILE
    data = pd.read_csv(path)
    times = list(data[data.iloc[:, 2] == country_name].iloc[:, 0])

    date_ref = datetime.date(2019, 12, 31)
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
