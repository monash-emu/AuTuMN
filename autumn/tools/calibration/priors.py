from typing import Tuple
from abc import ABC, abstractmethod


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


class BetaPrior(BasePrior):
    """
    A beta distributed prior.
    """

    def __init__(self, name: str, mean: float, ci: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.ci = ci
        self.name = name

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        return {
            **base_dict,
            "param_name": self.name,
            "distribution": "beta",
            "distri_mean": self.mean,
            "distri_ci": self.ci,
        }


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
