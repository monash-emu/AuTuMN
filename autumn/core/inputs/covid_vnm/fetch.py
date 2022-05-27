"""
This file is a placeholder for Vietnam and Ho Chi Minh City.
See Readme.md \data\inputs\covid_vnm for further details
"""


from autumn.settings import INPUT_DATA_PATH
from pathlib import Path
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

COVID_HCMC_TESTING_CSV = INPUT_DATA_PATH/ "covid_vnm"/ "testing.csv"


def fetch_covid_vnm_data():
    pass
