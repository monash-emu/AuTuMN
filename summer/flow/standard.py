from typing import List, Tuple, Dict, Callable
from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseTransitionFlow


class StandardFlow(BaseTransitionFlow):
    """
    A flow of people between compartments.
    """

    type = FlowType.STANDARD

    def __init__(
        self,
        source: Compartment,
        dest: Compartment,
        param_name: str,
        param_func: Callable[[str, float], float],
        adjustments: list = [],
    ):
        assert type(source) is Compartment
        assert type(dest) is Compartment
        self.adjustments = adjustments
        self.source = source
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func

    def get_net_flow(self, compartment_values, time):
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        return parameter_value * population

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return StandardFlow(
            source=kwargs["source"],
            dest=kwargs["dest"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            adjustments=kwargs["adjustments"],
        )

    def __repr__(self):
        return f"<StandardFlow from {self.source} to {self.dest} with {self.param_name}>"
