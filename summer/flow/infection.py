from typing import List, Tuple, Dict, Callable
from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseTransitionFlow


class BaseInfectionFlow(BaseTransitionFlow):
    def __init__(
        self,
        source: Compartment,
        dest: Compartment,
        param_name: str,
        param_func: Callable[[str, float], float],
        find_infectious_multiplier: Callable[[], float],
        adjustments: list = [],
    ):
        assert type(source) is Compartment
        assert type(dest) is Compartment
        self.adjustments = adjustments
        self.source = source
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func
        self.find_infectious_multiplier = find_infectious_multiplier

    def get_net_flow(self, compartment_values, time):
        multiplier = self.find_infectious_multiplier(self.source, self.dest)
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        return parameter_value * population * multiplier

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return self.__class__(
            source=kwargs["source"],
            dest=kwargs["dest"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            find_infectious_multiplier=self.find_infectious_multiplier,
            adjustments=kwargs["adjustments"],
        )


class InfectionDensityFlow(BaseInfectionFlow):
    type = FlowType.INFECTION_DENSITY

    def __repr__(self):
        return f"<InfectionDensityFlow from {self.source} to {self.dest} with {self.param_name}>"


class InfectionFrequencyFlow(BaseInfectionFlow):
    """
    A flow of people between compartments.
    """

    type = FlowType.INFECTION_FREQUENCY

    def __repr__(self):
        return f"<InfectionFrequencyFlow from {self.source} to {self.dest} with {self.param_name}>"
