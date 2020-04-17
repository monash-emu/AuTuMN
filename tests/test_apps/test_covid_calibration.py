from applications.covid_19.covid_calibration import *


def test_covid_calibration():
    country = 'australia'
    PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country)
    run_calibration_chain(30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc')


# def test_covid_calibration_free_start_time():
#     country = 'australia'
#     PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country)
#     del PAR_PRIORS[1]  # get rid of start time in estimated params
#
#     run_calibration_chain(30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode='lsm',
#                           _start_time_range=[-10, 0])


