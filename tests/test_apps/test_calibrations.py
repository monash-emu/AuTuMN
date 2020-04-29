import os

import pytest

from apps.marshall_islands import calibration as rmi_calibration
from apps.covid_19.calibration import base as covid_calibration


IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


@pytest.mark.skipif(not IS_GITHUB_CI, reason="This takes way too long to run locally.")
def test_marshall_islands_calibration():
    """
    Smoke test: ensure we can run the Marshall Islands calibration with nothing crashing.
    """
    rmi_calibration.run_calibration_chain(30, 1)


@pytest.mark.skipif(not IS_GITHUB_CI, reason="This takes way too long to run locally.")
def test_covid_calibration():
    country = "australia"
    PAR_PRIORS, TARGET_OUTPUTS = covid_calibration.get_priors_and_targets(country)
    covid_calibration.run_calibration_chain(
        30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc"
    )
