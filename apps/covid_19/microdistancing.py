"""
Extracts Nick Golding's survey CSV files from data\inputs\micro-distancing and
transforms it into a pandas data frame in preparation for loading
"""

import os
import pandas as pd
from autumn.constants import DATA_PATH

DATA_DIR = os.path.join(DATA_PATH, "inputs", "micro-distancing")

file_list = os.listdir(DATA_DIR)
file_list = [os.path.join(DATA_DIR, each) for each in file_list]

list_of_df = [pd.read_csv(each) for each in file_list]
df = pd.concat(list_of_df, axis=0, ignore_index=True)

df = df[
    ((df.state == "VIC") | (df.state == "Victoria"))
    & (df.question == "1.5m compliance")
    & (df.response == "Always")
]
df["micro-distancing"] = 1 - (df["count"] / df["respondents"])
