import pytest

from autumn.core.utils.tex_tools import (
    get_params_folder, 
    write_param_table_rows, 
    write_prior_table_rows,
)
from autumn.settings import Region, Models
from autumn.core.project import get_project

example_parameters = [
        "contact_rate",
        "ref_mixing_iso3",
        "infectious_seed",
        "asympt_infectiousness_effect",
        "isolate_infectiousness_effect",
        "testing_to_detection.assumed_tests_parameter",
        "testing_to_detection.assumed_cdr_parameter",
        "testing_to_detection.floor_value",
        "testing_to_detection.smoothing_period",    
        "booster_effect_duration",
        "immunity_stratification.infection_risk_reduction.high",
        "immunity_stratification.infection_risk_reduction.low", 
    ]


@pytest.mark.parametrize("model", [Models.SM_SIR])
@pytest.mark.parametrize("region", [Region.BHUTAN, Region.NCR])
@pytest.mark.parametrize("selected_parameters", [example_parameters])
def test_auto_params(model, region, selected_parameters):
    project = get_project(model, region)
    fixed_params_filename = get_params_folder(model, "bhutan", region, "auto_fixed_params")
    write_param_table_rows(fixed_params_filename, project, selected_parameters)


@pytest.mark.parametrize("model", [Models.SM_SIR])
@pytest.mark.parametrize("region", [Region.BHUTAN, Region.NCR])
def test_auto_priors(model, region):
    model = Models.SM_SIR
    region = Region.BHUTAN
    project = get_project(model, region)
    prior_param_filename = get_params_folder(model, "bhutan", region, "auto_priors")
    write_prior_table_rows(prior_param_filename, project)
