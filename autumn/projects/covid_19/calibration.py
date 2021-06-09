import numpy as np

from autumn.tools.calibration.priors import TruncNormalPrior


"""
Base COVID parameters

Rationale for the following two parameters described in parameters table of the methods Gdoc at:
https://docs.google.com/document/d/1Uhzqm1CbIlNXjowbpTlJpIphxOm34pbx8au2PeqpRXs/edit#
"""
EXPOSED_PERIOD_PRIOR = TruncNormalPrior(
    "sojourn.compartment_periods_calculated.exposed.total_period",
    mean=6,
    stdev=1,
    trunc_range=[1.0, np.inf],
)
ACTIVE_PERIOD_PRIOR = TruncNormalPrior(
    "sojourn.compartment_periods_calculated.active.total_period",
    mean=6.5,
    stdev=0.77,
    trunc_range=[4.0, np.inf],
)
COVID_GLOBAL_PRIORS = [EXPOSED_PERIOD_PRIOR, ACTIVE_PERIOD_PRIOR]
