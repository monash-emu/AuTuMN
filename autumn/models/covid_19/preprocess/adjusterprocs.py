from typing import List
import numpy as np

from summer.adjust import AdjustmentSystem


class AbsPropIsolatedSystem(AdjustmentSystem):
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """
    def __init__(self, early_rate: float):
        """Initialize the system

        Args:
            early_rate (float): Rate by which all output is scaled
        """
        self.early_rate = early_rate

    def prepare_to_run(self, component_data: List[dict]):
        """Compile all components into arrays used for fast computation

        Args:
            component_data (List[dict]): List containing data specific to individual flow adjusments
        """

        self.proportion_sympt = np.empty_like(component_data, dtype=float)
        self.proportion_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.proportion_sympt[i] = component['proportion_sympt']
            self.proportion_hosp[i] = component['proportion_hosp']

    def get_weights_at_time(self, time, computed_values):
        cdr = computed_values["cdr"]
        return get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr) * self.early_rate


class AbsPropSymptNonHospSystem(AdjustmentSystem):
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """
    def __init__(self, early_rate: float):
        """Initialize the system

        Args:
            early_rate (float): Rate by which all output is scaled
        """
        self.early_rate = early_rate

    def prepare_to_run(self, component_data: List[dict]):
        """Compile all components into arrays used for fast computation

        Args:
            component_data (List[dict]): List containing data specific to individual flow adjusments
        """

        self.proportion_sympt = np.empty_like(component_data, dtype=float)
        self.proportion_hosp = np.empty_like(component_data, dtype=float)

        for i, component in enumerate(component_data):
            self.proportion_sympt[i] = component['proportion_sympt']
            self.proportion_hosp[i] = component['proportion_hosp']

    def get_weights_at_time(self, time, computed_values):
        cdr = computed_values["cdr"]
        prop_isolated = get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr)
        prop_sympt_non_hospital = self.proportion_sympt - self.proportion_hosp - prop_isolated
        return prop_sympt_non_hospital * self.early_rate


def get_abs_prop_isolated(proportion_sympt, proportion_hosp, cdr) -> np.ndarray:
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    
    Args:
        proportion_sympt ([type]): Float or np.ndarray
        proportion_hosp ([type]): Float or np.ndarray
        cdr ([type]): Float

    Returns:
        [np.ndarray]: Output value
    """
    target_prop_detected = proportion_sympt * cdr
    return np.maximum(0., target_prop_detected - proportion_hosp)
