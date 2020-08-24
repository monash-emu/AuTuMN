from typing import List, Tuple, Dict, Callable

import numpy as np

from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseEntryFlow


class ImportFlow(BaseEntryFlow):
    """
    Calculates importation, based on the current population size.
    TODO: Remove population component
    """

    type = FlowType.IMPORT

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
        return self.get_weight_value(time)

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return ImportFlow(
            dest=kwargs["dest"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            adjustments=kwargs["adjustments"],
        )

    def __repr__(self):
        return f"<ImportFlow to {self.dest} with {self.param_name}>"

