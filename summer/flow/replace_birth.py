from typing import List, Tuple, Dict, Callable
from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .base import BaseEntryFlow


class ReplacementBirthFlow(BaseEntryFlow):
    """
    Calculates births by replacing total deaths.
    """

    type = FlowType.BIRTH

    def __init__(self, dest: Compartment, get_total_deaths: Callable[[], float], adjustments=[]):

        assert type(dest) is Compartment
        self.adjustments = adjustments
        self.param_name = "total_deaths"
        self.dest = dest
        self.get_total_deaths = get_total_deaths

    def param_func(self, name, time):
        """
        Hack so that `get_weight_value` will use `get_total_deaths`
        and also apply any adjustments.
        """
        assert time is None, "Total deaths is not a function of time."
        if name == "total_deaths":
            return self.get_total_deaths()
        else:
            raise ValueError("Cannot use param_func for replace deaths.")

    def get_net_flow(self, compartment_values, time):
        return self.get_weight_value(None)

    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        return ReplacementBirthFlow(
            dest=kwargs["dest"],
            param_name=kwargs["param_name"],
            param_func=kwargs["param_func"],
            adjustments=kwargs["adjustments"],
        )

    def __repr__(self):
        return f"<ReplacementBirthFlow to {self.dest} with {self.param_name}>"
