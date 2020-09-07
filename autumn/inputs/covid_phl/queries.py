import pandas as pd
from datetime import datetime
import numpy as np

from .fetch import COVID_PHL_CSV_PATH
from .preprocess.py import facilities
from .preprocess.py import regions

COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

def get_phl_testing_numbers(region):
    # read csv
    testdf = pd.read_csv(COVID_PHL_CSV_PATH) 
    # format date and convert to times since Dec. 31, 2019
    testdf['report_date'] = pd.to_datetime(testdf['report_date'])
    testdf['times'] = testdf.report_date - COVID_BASE_DATETIME
    testdf['times'] = testdf['times'] / np.timedelta64(1, 'D')
    if (region == 'philippines'):
        regional_tests = testdf
    else: 
        # subset to specific region
        regional_facilities = [i for i in range(0,len(facilities)) if regions[i] == region]
        regional_tests = testdf[testdf['facility_name'].isin(facilities[min(regional_facilities):max(regional_facilities)])]
    # sum unique individual tests per day
    regional_tests_sub = regional_tests.groupby('times', as_index = False)[['daily_output_unique_individuals']].sum()   
    # return results
    test_dates = regional_tests_sub['times']
    test_values = regional_tests_sub['daily_output_unique_individuals']
    return test_dates, test_values
