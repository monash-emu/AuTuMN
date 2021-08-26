from typing import Tuple, List
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from scipy import stats

from autumn.tools.calibration.utils import calculate_prior, specify_missing_prior_params

class BasePrior(ABC):
    def __init__(self, sampling: str = None, jumping_stdev: float = None):
        self.sampling = sampling
        self.jumping_stdev = jumping_stdev

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns the prior as a dict... for now"""
        prior = {}
        if self.jumping_stdev:
            prior["jumping_stdev"] = self.jumping_stdev

        if self.sampling:
            prior["sampling"] = self.sampling

        return prior

    @abstractmethod
    def calculate(self, x: float, log=True) -> float:
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_finite_range(self) -> Tuple[float, float]:
        pass

class BetaPrior(BasePrior):
    """
    A beta distributed prior.
    """

    def __init__(self, name: str, mean: float, ci: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.ci = ci
        self.name = name

        # Bit of a hack to leverage existing functionality
        # FIXME It's a mess, sort it out

        self.a = None
        self.b = None

        _temp_dict = self.to_dict()
        specify_missing_prior_params([_temp_dict])
        self.a = _temp_dict['distri_params'][0]
        self.b = _temp_dict['distri_params'][1]

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        return {
            **base_dict,
            "param_name": self.name,
            "distribution": "beta",
            "distri_mean": self.mean,
            "distri_ci": self.ci,
        }

    def calculate(self, x: float, log=True) -> float:
        if log:
            y = stats.beta.logpdf(x, self.a, self.b)
        else:
            y = stats.beta.pdf(x, self.a, self.b)
        return float(y)

    def get_bounds(self) -> Tuple[float, float]:
        return 0.0, 1.0

    def get_finite_range(self) -> Tuple[float, float]:
        return stats.beta.ppf((0.025, 0.975), self.a, self.b)


class UniformPrior(BasePrior):
    """
    A uniformily distributed prior.
    """

    def __init__(self, name: str, domain: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.start, self.end = domain
        self.name = name

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        return {
            **base_dict,
            "param_name": self.name,
            "distribution": "uniform",
            "distri_params": [self.start, self.end],
        }

    def calculate(self, x, log=True) -> float:
        y = 1.0 / (self.end - self.start)
        if log:
            y = np.log(y)

        return y

    def get_bounds(self) -> Tuple[float, float]:
        return self.start, self.end

    def get_finite_range(self) -> Tuple[float, float]:
        return self.start, self.end

class TruncNormalPrior(BasePrior):
    """
    A prior with a truncated normal distribution.
    """

    def __init__(
        self, name: str, mean: float, stdev: float, trunc_range: Tuple[float, float], **kwargs
    ):
        super().__init__(**kwargs)
        self.mean, self.stdev = mean, stdev
        self.name = name
        self.trunc_range = trunc_range

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        return {
            **base_dict,
            "param_name": self.name,
            "distribution": "trunc_normal",
            "distri_params": [self.mean, self.stdev],
            "trunc_range": self.trunc_range,
        }

    def calculate(self, x: float, log=True) -> float:
        mu, sd = self.mean, self.stdev
        bounds = self.trunc_range
        if log:
            y = stats.truncnorm.logpdf(
                x, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
                )
        else:
            y = stats.truncnorm.pdf(
                x, (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
                )
        return float(y)

    def get_bounds(self) -> Tuple[float, float]:
        return self.trunc_range

    def get_finite_range(self) -> Tuple[float, float]:
        mu = self.mean
        sd = self.stdev
        bounds = self.trunc_range
        return stats.truncnorm.ppf(
            (0.025, 0.975), (bounds[0] - mu) / sd, (bounds[1] - mu) / sd, loc=mu, scale=sd
        )


class BasePriorSet(ABC):
    """A collection of priors for a given set of parameters
    """
    def __init__(self, priors: List[BasePrior]):
        self._dict_priors = [p.to_dict() for p in priors]
        self.priors = OrderedDict([(p.name, p) for p in priors ])

    @abstractmethod
    def logprior(self, params: dict):
        pass

    def get(self, k:str) -> BasePrior:
        return self.priors.get(k)

    def as_list_of_dicts(self):
        return [p.to_dict() for p in self.priors.values()]

class IndependentPriorSetClassic(BasePriorSet):

    def logprior(self, params: dict):
        logp = 0.0
        for param_name, value in params.items():
            prior_dict = [d for d in self._dict_priors if d["param_name"] == param_name][0]
            logp += calculate_prior(prior_dict, value, log=True)

        return logp

class IndependentPriorSet(BasePriorSet):

    def logprior(self, params: dict):
        logp = 0.0
        for param_name, value in params.items():
            logp += self.priors[param_name].calculate(value, log=True)

        return logp
