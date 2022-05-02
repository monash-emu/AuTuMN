import pytest
from autumn.models.covid_19.strat_processing.vaccination import find_vaccine_action

VACCINE_PARAM_TESTS = [
    # overall_eff, vacc_prevention, infection_efficacy, severity_efficacy
    (0.1, 0.5, 0.05, 0.053),
    (0.77, 0.2, 0.154, 0.728),
    (0, 0, 0, 0),
    (0.3, 0.65, 0.195, 0.130),
    (0.98, 0.76, 0.745, 0.922),
    (0.01, 0.75, 0.007, 0.003),
    (0.78, 0.99, 0.772, 0.034),
    (1, 1, 1, 0),
]


@pytest.mark.parametrize(
    "overall_eff, vacc_prevention, infection_efficacy, severity_efficacy", VACCINE_PARAM_TESTS
)
def test_vaccine_params(overall_eff, vacc_prevention, infection_efficacy, severity_efficacy):
    """
    Checking that converted parameters end up in the correct range, given sensible requests.

    """
    actual_infection_efficacy, actual_severity_efficacy = find_vaccine_action(
        vacc_prevention, overall_eff
    )
    assert round(actual_infection_efficacy, 3) == infection_efficacy
    assert round(actual_severity_efficacy, 3) == severity_efficacy
