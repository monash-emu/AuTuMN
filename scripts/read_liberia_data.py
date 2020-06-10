import pandas as pd

file_path = "../data/xls/data_liberia.csv"

data = pd.read_csv(file_path, sep=',')

_new_cases = {
    'times': list(data['time']),
    'values': list(data['new_cases']),
}

_new_deaths = {
    'times': list(data['time']),
    'values': list(data['new_deaths']),
}

for i, data_dict in enumerate([_new_cases, _new_deaths]):

    print("#############")
    print(data_dict['times'])
    print(data_dict['values'])
    print([[x] for x in data_dict['values']])
