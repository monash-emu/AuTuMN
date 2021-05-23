"""
Entry point for PyCharm users to run an application
"""
import os

from apps import covid_19, sir_example, tuberculosis, tuberculosis_strains
from autumn.region import Region

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
os.chdir("..")  # Make repo root the current directory


class Mode:
    RUN = "run"
    CALIBRATE = "calibrate"


class App:
    COVID_19 = "covid_19"
    TUBERCULOSIS = "tuberculosis"
    TUBERCULOSIS_STRAINS = "tuberculosis_strains"
    SIR_EXAMPLE = "sir_example"


def run(app, regions, mode, run_scenarios=False, calibration_time=None):
    for region in regions:
        if app == App.COVID_19:
            app_region = covid_19.app.get_region(region)
        elif app == App.TUBERCULOSIS:
            app_region = tuberculosis.app.get_region(region)
        elif app == App.TUBERCULOSIS_STRAINS:
            app_region = tuberculosis_strains.app.get_region(region)
        elif app == App.SIR_EXAMPLE:
            app_region = sir_example.app.get_region(region)
        else:
            msg = f"The requested app {app} does not exist or has not been imported."
            raise ValueError(msg)

        if mode == Mode.RUN:
            app_region.run_model(run_scenarios=run_scenarios)
        elif mode == Mode.CALIBRATE:
            app_region.calibrate_model(max_seconds=calibration_time, run_id=0, num_chains=1)
        else:
            msg = f"The requested mode {mode} is not supported."
            raise ValueError(msg)


##################################
#       User's configuration
_app = App.COVID_19
_regions = [Region.MALAYSIA]  # e.g. [Region.MALAYSIA],  OPTI_REGIONS, Region.MALAYSIA_REGIONS, Region.PHILIPPINES_REGIONS
_mode = Mode.RUN
_run_scenarios = False
_calibration_time = 5  # calibration duration (only required for calibration mode)

# run the models
run(_app, _regions, _mode, _run_scenarios, _calibration_time)
