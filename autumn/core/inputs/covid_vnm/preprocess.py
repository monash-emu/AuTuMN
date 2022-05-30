import pandas as pd

from autumn.core.db import Database

from .fetch import COVID_HCMC_TESTING_CSV

def preprocess_covid_vnm(input_db: Database):

    df = pd.read_csv(COVID_HCMC_TESTING_CSV)
    input_db.dump_df("covid_vnm", df)