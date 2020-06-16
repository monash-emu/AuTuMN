import pytest

from apps.marshall_islands import calibration as rmi_calibration
from apps.covid_19 import calibration as covid_calibration


CALIBRATION_REGIONS = list(covid_calibration.CALIBRATIONS.keys())


@pytest.mark.calibrate_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", CALIBRATION_REGIONS)
def test_covid_calibration(region):
    calib_func = covid_calibration.get_calibration_func(region)
    calib_func(15, 0)
