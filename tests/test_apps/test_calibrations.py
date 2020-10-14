import autumn
import pytest

from apps import covid_19, tuberculosis


@pytest.mark.calibrate_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_covid_calibration(region):
    """
    Calibration smoke test - make sure everything can run for 10 seconds without exploding.
    """
    region_app = covid_19.app.get_region(region)
    region_app.calibrate_model(10, 1, 1)


@pytest.mark.calibrate_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", tuberculosis.app.region_names)
def test_tuberculosis_calibration(region):
    """
    Calibration smoke test - make sure everything can run for 15 seconds without exploding.
    """
    region_app = tuberculosis.app.get_region(region)
    region_app.calibrate_model(10, 1, 1)
