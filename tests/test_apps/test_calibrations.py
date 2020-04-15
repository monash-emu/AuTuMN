from applications.marshall_islands import calibration as rmi_calibration
from applications.covid_19.calibration import base as covid_calibration


def test_marshall_islands_calibration():
    """
    Smoke test: ensure we can run the Marshall Islands calibration with nothing crashing.
    """
    rmi_calibration.run_calibration_chain(30, 1)


def test_covid_calibration():
    country = "australia"
    PAR_PRIORS, TARGET_OUTPUTS = covid_calibration.get_priors_and_targets(country)
    covid_calibration.run_calibration_chain(
        30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc"
    )


# def test_covid_calibration_free_start_time():
#     country = 'australia'
#     PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country)
#     del PAR_PRIORS[1]  # get rid of start time in estimated params
#
#     run_calibration_chain(30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode='lsm',
#                           _start_time_range=[-10, 0])
