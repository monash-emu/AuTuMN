from typing import List, Any

from summer.adjust import AdjustmentComponent, AdjustmentSystem
import numpy as np

class AbsPropIsolatedProc:
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """
    def __init__(self, age_idx, abs_props, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.early_rate = early_rate

    def __call__(self, time, computed_values):
        return get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, computed_values["cdr"]) * self.early_rate

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

class AbsPropSymptNonHospProc:
    """
    Returns the absolute proportion of infected not entering the hospital.
    This also does not count people who are isolated/detected.
    """
    def __init__(self, age_idx, abs_props, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.early_rate = early_rate

    def get_abs_prop_sympt_non_hospital(self, time, cdr):
        prop_isolated = get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr)
        return self.proportion_sympt - self.proportion_hosp - prop_isolated

    def __call__(self, time, computed_values):
        return self.get_abs_prop_sympt_non_hospital(time, computed_values["cdr"]) * self.early_rate

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
    """    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    
    Args:
        proportion_sympt ([type]): Float or np.ndarray
        proportion_hosp ([type]): Float or np.ndarray
        cdr ([type]): Float

    Returns:
        [np.ndarray]: Output value
    """
    target_prop_detected = proportion_sympt * cdr
    return np.maximum(0.0, target_prop_detected - proportion_hosp)
