import os

import pandas as pd

from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import (
    DURATIONS,
    MODES,
    OBJECTIVES,
    objective_function,
    run_root_model,
)
from autumn.projects.covid_19.mixing_optimisation.utils import (
    get_scenario_mapping,
    get_scenario_mapping_reverse,
)
from autumn.coreimport db
from autumn.settings import BASE_PATH


def load_derived_output(database_path, output_name):
    df = db.load.load_derived_output_tables(database_path, output_name)[0]
    return df
