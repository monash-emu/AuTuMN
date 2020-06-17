import pandas as pd
import copy

deaths_file_path = "calibration_deaths.csv"
cases_file_path = "calibration_cases.csv"

deaths_data = pd.read_csv(deaths_file_path, sep=",")
cases_data = pd.read_csv(cases_file_path, sep=",")

deaths = {
    "times": list(deaths_data["deaths_times"]),
    "values": list(deaths_data["deaths_values"]),
}

cases = {
    "times": list(cases_data["cases_times"]),
    "values": list(cases_data["cases_values"]),
}

print("Deaths:")
print(deaths["times"])
print([[d] for d in deaths["values"]])

[print() for _ in range(2)]
print("Cases:")
print(cases["times"])
print([[c] for c in cases["values"]])
