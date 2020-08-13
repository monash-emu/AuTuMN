from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable

import numpy as np

from summer.stratification import Stratification
from summer.constants import FlowAdjustment


class BaseFlow(ABC):
    """
    Abstract base class for all flows.
    A flow represents the movement of people from one compartment to another.
    """

    param_name = None
    param_func = None
    adjustments = None

    @property
    @abstractmethod
    def type(self):
        """Returns the type of flow"""
        pass

    def get_weight_value(self, time: float):
        """
        Returns the flow's weight at a given time.
        Applies any stratification adjustments to the base parameter.
        """
        value = self.param_func(self.param_name, time)
        return self._apply_adjustments(value, time)

    def _apply_adjustments(self, value: float, time: float):
        for adjust_type, adjust_value in self.adjustments:
            if adjust_type == FlowAdjustment.COMPOSE:
                # Multiply by time variant function value.
                value *= self.param_func(adjust_value, time)
            elif adjust_type == FlowAdjustment.OVERWRITE:
                # Overwrite with adjustment value.
                value = adjust_value
            elif adjust_type == FlowAdjustment.MULTIPLY:
                # Multiply by adjustment value.
                value *= adjust_value

        return value

    @abstractmethod
    def update_compartment_indices(self, mapping: Dict[str, float]):
        """
        Update index which maps flow compartments to compartment value array.
        """
        pass

    @abstractmethod
    def get_net_flow(self, compartment_values: np.ndarray, time: float) -> float:
        """
        Returns the net flow value at a given time.
        """
        pass

    @abstractmethod
    def stratify(self, strat: Stratification) -> list:
        """
        Returns a list of new, stratified flows to replace the current flow.
        """
        pass

    @abstractmethod
    def copy(self, **kwargs):
        """
        Creates a modified copy of the flow for stratification.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Returns a text representation of the flow.
        """
        pass
