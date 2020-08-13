from typing import List, Tuple, Dict, Callable

import numpy as np
from numba import jit

from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseEntryFlow


class CrudeBirthFlow(BaseEntryFlow):
    """
    Calculates births using a 'crude birth rate' method.
    """

    type = FlowType.BIRTH

    def __init__(
        self,
        dest: Compartment,
        param_name: str,
        param_func: Callable[[str, float], float],
        adjustments: list = [],
    ):
        assert type(dest) is Compartment
        self.adjustments = adjustments
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func

    def get_net_flow(self, compartment_values, time):
        parameter_value = self.get_weight_value(time)
        population = _find_sum(compartment_values)
        return parameter_value * population

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return CrudeBirthFlow(
            dest=kwargs["dest"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            adjustments=kwargs["adjustments"],
        )

    def __repr__(self):
        return f"<CrudeBirthFlow to {self.dest} with {self.param_name}>"


@jit(nopython=True)
def _find_sum(compartment_values: np.ndarray) -> float:
    return compartment_values.sum()

