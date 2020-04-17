from applications.covid_19.covid_calibration import *


def test_covid_calibration():
    country = 'australia'
    PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country)
    run_calibration_chain(30, 1, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc')
