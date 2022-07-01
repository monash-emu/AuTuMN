from autumn.core.project import get_project, ParameterSet
from matplotlib import pyplot as plt
from autumn.core.plots.utils import REF_DATE, change_xaxis_to_date
import pandas as pd
from autumn.core.inputs.demography.queries import get_population_by_agegroup
iso3s = ["malaysia", "national-capital-region"]

for iso3 in iso3s:
    project = get_project("sm_sir", iso3)
    # run baseline model
    model_0 = project.run_baseline_model(project.param_set.baseline)
