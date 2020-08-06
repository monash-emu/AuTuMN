import pandas as pd
import os


def get_vic_testing_numbers():
    # This is such a hack and also note that it requires pre-processing of the Excel file to ensure data are in a
    # numeric form before they are read.
    path = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(path, "vic_testing.csv"))
    days_from_excel_date_to_our_date = 43831
    dates = [i_date - days_from_excel_date_to_our_date for i_date in data.iloc[:, 0]]
    values = list(data.iloc[:, 1])
    return dates, values
