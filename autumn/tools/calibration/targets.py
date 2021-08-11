from typing import List, Tuple
from abc import ABC, abstractmethod

from autumn.tools.project.timeseries import TimeSeries
from .priors import UniformPrior


SOMETHING = "something"


def convert_targets_to_dict(raw_targets):
    """
    Take the raw targets data structure, which is a list that you have to dig the targets out of
    and convert it into a dictionary with the names of the targets being the keys.

    """
    targets_dict = {}
    for i_target in range(len(raw_targets)):
        output_key = raw_targets[i_target]["output_key"]
        targets_dict[output_key] = {
            "times": raw_targets[i_target]["years"],
            "values": raw_targets[i_target]["values"]
        }
    return targets_dict


class BaseTarget(ABC):

    timeseries: TimeSeries

    def __init__(self, time_weights: List[float] = None):
        self.time_weights = time_weights

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns the target as a dict... for now"""
        target = {}
        if self.time_weights:
            target["time_weights"] = self.time_weights

        return target


class PoissonTarget(BaseTarget):
    """
    A calibration target sampled from a Poisson distribution
    """

    def __init__(self, timeseries: TimeSeries, **kwargs):
        super().__init__(**kwargs)
        self.timeseries = timeseries

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        return {
            **base_dict,
            "output_key": self.timeseries.name,
            "years": self.timeseries.times,
            "values": self.timeseries.values,
            "loglikelihood_distri": "poisson",
        }


class NegativeBinomialTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(self, timeseries: TimeSeries, dispersion_param: float = None, **kwargs):
        super().__init__(**kwargs)
        self.timeseries = timeseries
        self.dispersion_param = dispersion_param

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        target = {
            **base_dict,
            "output_key": self.timeseries.name,
            "years": self.timeseries.times,
            "values": self.timeseries.values,
            "loglikelihood_distri": "negative_binomial",
        }
        if self.dispersion_param:
            target['dispersion_param'] = self.dispersion_param

        return target


class TruncNormalTarget(BaseTarget):
    """
    A calibration target sampled from a truncated normal distribution
    """

    def __init__(
        self,
        timeseries: TimeSeries,
        trunc_range: Tuple[float, float],
        stdev: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timeseries = timeseries
        self.trunc_range = trunc_range
        self.stdev = stdev

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        target = {
            **base_dict,
            "output_key": self.timeseries.name,
            "years": self.timeseries.times,
            "values": self.timeseries.values,
            "trunc_range": self.trunc_range,
            "loglikelihood_distri": "trunc_normal",
        }
        if self.stdev:
            target["sd"] = self.stdev

        return target


class NormalTarget(BaseTarget):
    """
    A calibration target sampled from a normal distribution
    """

    def __init__(self, timeseries: TimeSeries, stdev: float = None, **kwargs):
        super().__init__(**kwargs)
        self.timeseries = timeseries
        self.stdev = stdev

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        target = {
            **base_dict,
            "output_key": self.timeseries.name,
            "years": self.timeseries.times,
            "values": self.timeseries.values,
            "loglikelihood_distri": "normal",
        }
        if self.stdev:
            target["sd"] = self.stdev

        return target


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

        max_val = max(target.timeseries.values)
        # sd_ that would make the 95% gaussian CI cover half of the max value (4*sd = 95% width)
        sd_ = 0.25 * max_val / 4.0
        lower_sd = sd_ / 2.0
        upper_sd = 2.0 * sd_
        name = f"{target.timeseries.name}_dispersion_param"
        prior = UniformPrior(name, [lower_sd, upper_sd])
        priors.append(prior)

    return priors
