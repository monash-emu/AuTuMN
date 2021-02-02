import pandas as pd
import os
from autumn import db

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import (
    MODES,
    DURATIONS,
    OBJECTIVES,
    run_root_model,
    objective_function,
)
from settings import BASE_PATH
from apps.covid_19.mixing_optimisation.utils import (
    get_scenario_mapping,
    get_scenario_mapping_reverse,
)


def load_derived_output(database_path, output_name):
    df = db.load.load_derived_output_tables(database_path, output_name)[0]
    return df
