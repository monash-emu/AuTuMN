import os
import pandas as pd
from autumn.constants import DATA_PATH
import datetime


HOSPITAL_DATA_DIR = os.path.join(DATA_PATH, "hospitalisation_data")
country_mapping = {"united-kingdom": "The United Kingdom"}


def read_hospital_data_from_csv(variable="hospital_occupancy", country="belgium", data_start_time=61, data_end_time=152):
    """
    Read hospital data from file 'hospital_data_europe.csv'
    :param variable: one of 'hospital_occupancy', 'hospital_admission', 'icu_occupancy', 'icu_admission'
    :param country: country
    """
    if country in country_mapping:
        country_name = country_mapping[country]
    else:
        country_name = country.title()
    filename = f"hospital_data_europe.csv"
    path = os.path.join(HOSPITAL_DATA_DIR, filename)
    data = pd.read_csv(path)

    column_name = country + "_" + variable
    mask_1 = data['time'] >= data_start_time
    mask_2 = data['time'] <= data_end_time
    mask_3 = pd.notnull(data[column_name])
    mask = [m_1 and m_2 and m_3 for (m_1, m_2, m_3) in zip(mask_1, mask_2, mask_3)]
    times = [float(t) for t in data[mask]['time']]
    values = [float(v) for v in data[mask][column_name]]

    return times, values
