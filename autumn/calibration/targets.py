from typing import List, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from scipy import stats

from summer.utils import ref_times_to_dti

#from autumn.core.project.timeseries import TimeSeries
from .priors import UniformPrior


class BaseTarget(ABC):

    data: pd.Series

    def __init__(self, data: pd.Series, time_weights: np.ndarray = None):
        # Make things easier for calibration by sanitizing the data here
        self.data = data.dropna()
        if time_weights is not None:
            self.time_weights = np.array(time_weights)
        else:
            self.time_weights = None
        self.stdev = None
        self.cis = None


class PoissonTarget(BaseTarget):
    """
    A calibration target sampled from a Poisson distribution
    """

    def __init__(self, data:pd.Series, **kwargs):
        super().__init__(data, **kwargs)
        self.loglikelihood_distri = "poisson"


class NegativeBinomialTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(self, data: pd.Series, dispersion_param: float = None, **kwargs):
        super().__init__(data, **kwargs)
        self.dispersion_param = dispersion_param
        self.loglikelihood_distri = "negative_binomial"


class TruncNormalTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(
        self,
        data: pd.Series,
        trunc_range: Tuple[float, float],
        stdev: float = None,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.trunc_range = trunc_range
        self.stdev = stdev
        self.loglikelihood_distri = "trunc_normal"


class NormalTarget(BaseTarget):
    """
    A calibration target sampled from a normal distribution
    """

    def __init__(self, data: pd.Series, stdev: float = None, **kwargs):
        super().__init__(data, **kwargs)
        self.stdev = stdev
        self.loglikelihood_distri = "normal"


def get_dispersion_priors_for_gaussian_targets(targets: List[BaseTarget]):
    """
    Returns any dispersion priors to be used alongside the targets.

    The dispersion parameter defines how fussy we want to be about capturing data with the model.
    If its value is tiny, this means we are using a likelihood function that is very skewed and that will reject any model run that is not fitting the data perfectly well.
    Conversely, a large value will allow for significant discrepancies between model predictions and data.
    """
    priors = []
    for target in targets:
        if type(target) not in [TruncNormalTarget, NormalTarget]:
            continue
        if target.stdev is not None:
            continue

        max_val = max(target.data)
        # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
        sd_ = 0.25 * max_val / 4.0
        lower_sd = sd_ / 2.0
        upper_sd = 2.0 * sd_
        name = f"{target.data.name}_dispersion_param"
        prior = UniformPrior(name, [lower_sd, upper_sd])
        priors.append(prior)

    return priors

def truncnormal_logpdf(target_data: np.ndarray, model_output: np.ndarray, trunc_vals: Tuple[float, float], sd: float):
    """
    Return the logpdf of a truncated normal target, with scaling transforms
    according to:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """
    a, b = (trunc_vals[0] - model_output) / sd, (trunc_vals[1] - model_output) / sd
    return stats.truncnorm.logpdf(x=target_data, a=a, b=b, loc=model_output, scale=sd)
