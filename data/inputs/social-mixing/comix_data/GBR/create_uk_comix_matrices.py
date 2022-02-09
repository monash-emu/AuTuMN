import pandas as pd
import os
import numpy as np
from autumn.settings.folders import INPUT_DATA_PATH

"""
Read and format original data
"""
data_path = os.path.join(INPUT_DATA_PATH, "social-mixing", "comix_data", "GBR", "contact_matrices_9_periods.csv")
data = pd.read_csv(data_path, index_col="ID")
data.replace(["5-Nov", "Dec-17"], ["5-11", "12-17"], inplace=True)
agegroups = data["Participant age"][0:9].to_list()
formated_agegroups = [f"[{a.split('-')[0]},{int(a.split('-')[1]) + 1})" for a in agegroups[:-1]] + ["70+"]
periods = list(data["Period"].unique())

"""
Create a mixing matrix for each of the nine comix periods
"""
n = len(agegroups)
for period in periods:
    m = np.zeros((n, n))
    filename = f"comix_matrix_period_{period[0]}.csv"
    output_path = os.path.join(INPUT_DATA_PATH, "social-mixing", "comix_data", "GBR", filename)
    for i, age_contactor in enumerate(agegroups):
        for j, age_contactee in enumerate(agegroups):
            m[i, j] = float(data[
                (data["Participant age"] == age_contactor) &
                (data["Contact age"] == age_contactee) &
                (data["Period"] == period)
            ]["mean contacts"])

    pd.DataFrame(m, index=formated_agegroups).to_csv(output_path, header=formated_agegroups)


