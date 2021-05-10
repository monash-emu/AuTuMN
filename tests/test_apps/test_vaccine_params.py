from copy import deepcopy
import pytest

from apps import covid_19
from autumn.utils.utils import merge_dicts
from apps.covid_19.model.preprocess.vaccination import add_vaccine_infection_and_severity


PARAMETER_VALUES = \
    [0., 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99, 1]


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_vaccine_severity_infection_parameters(region):
    """
    Check that requests are being submitted correctly for all regions and all scenarios.

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


@pytest.mark.parametrize("vacc_prevention", PARAMETER_VALUES)
@pytest.mark.parametrize("overall_eff", PARAMETER_VALUES)
def test_vaccine_params(overall_eff, vacc_prevention):
    """
    Checking that converted parameters end up in the correct range, given sensible requests.

    """
    infection_efficacy, severity_efficacy = add_vaccine_infection_and_severity(vacc_prevention, overall_eff)
    assert infection_efficacy <= 1, "Should belong to [0, 1]"
    assert infection_efficacy >= 0, "Should belong to [0, 1]"
    assert severity_efficacy <= 1, "Should belong to [0, 1]"
    assert severity_efficacy >= 0, "Should belong to [0, 1]"
    assert infection_efficacy >= 0, "Should belong to [0, Ve]"
    assert infection_efficacy <= overall_eff, "Should belong to [0, Ve]"
