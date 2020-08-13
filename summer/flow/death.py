from typing import List, Tuple, Dict, Callable
from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseExitFlow


class BaseDeathFlow(BaseExitFlow):
    """A flow representing deaths"""

    def __init__(
        self,
        source: Compartment,
        param_name: str,
        param_func: Callable[[str, float], float],
        adjustments: list = [],
    ):
        assert type(source) is Compartment
        self.adjustments = adjustments
        self.source = source
        self.param_name = param_name
        self.param_func = param_func

    def get_net_flow(self, compartment_values, time):
        parameter_value = self.get_weight_value(time)
        population = compartment_values[self.source.idx]
        flow_rate = parameter_value * population
        return flow_rate

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return self.__class__(
            source=kwargs["source"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            adjustments=kwargs["adjustments"],
        )


class InfectionDeathFlow(BaseDeathFlow):
    """A flow representing infection deaths"""

    type = FlowType.DEATH

    def __repr__(self):
        return f"<DeathFlow from {self.source} with {self.param_name}>"


class UniversalDeathFlow(BaseDeathFlow):
    """A flow representing non-infection deaths"""

    type = FlowType.UNIVERSAL_DEATH

    def __repr__(self):
        return f"<UniversalDeathFlow from {self.source} with {self.param_name}>"
