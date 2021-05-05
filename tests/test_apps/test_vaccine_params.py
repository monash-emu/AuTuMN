from copy import deepcopy

import pytest
from summer import CompartmentalModel
from summer.legacy.model import StratifiedModel

from apps import covid_19
from autumn.utils.utils import merge_dicts

# from apps.covid_19.model.preprocess.vaccination import add_vaccine_infection_and_severity
@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.app.region_names)

def test_vaccine_severity_infection_parameters(region):
    """checking if overall vaccine efficacy is 0
    """
    region_app = covid_19.app.get_region(region)
    for idx, scenario_params in enumerate(region_app.params["scenarios"].values()):
        default_params = deepcopy(region_app.params["default"])
        params = merge_dicts(scenario_params, default_params)

        if params["vaccination"] is not None:
            assert params["vaccination"]["overall_efficacy"] <= 1, "Should belong to [0,1]"
            assert params["vaccination"]["overall_efficacy"] >= 0, "Should belong to [0,1]"
            assert params["vaccination"]["vacc_prop_prevent_infection"] <= 1, "Should belong to [0,1]"
            assert params["vaccination"]["vacc_prop_prevent_infection"] >= 0, "Should belong to [0,1]"

            # infection_efficacy, severity_efficacy = add_vaccine_infection_and_severity(
            #     params["vaccination"]["vacc_prop_prevent_infection"], params["vaccination"]["overall_efficacy"])
            # assert infection_efficacy >= 0, "Should belong to [0,Ve]"
            # assert infection_efficacy <= params["vaccination"]["overall_efficacy"] , "Should belong to [0,Ve]"