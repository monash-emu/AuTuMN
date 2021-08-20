import numpy as np
import numba

from autumn.models.covid_19.constants import (
    Compartment,
    INFECTIOUS_COMPARTMENTS,
    Clinical,
    CLINICAL_STRATA,
    NOTIFICATION_CLINICAL_STRATA
)

from summer.compute import ComputedValueProcessor, find_sum


def get_tracing_param(assumed_trace_prop, assumed_prev):
    """
    Calculate multiplier for the relationship between traced proportion and prevalence for use in the next function.
    """
    assert 0. <= assumed_trace_prop <= 1.001
    assert 0. <= assumed_prev <= 1.
    return -np.log(assumed_trace_prop) / assumed_prev


def get_traced_prop(trace_param, prev):
    """
    Function for the proportion of detected people who are traced.
    """
    return np.exp(-prev * trace_param)


def contact_tracing_func(time, computed_values):
    """
    Multiply the flow rate through by the source compartment to get the final absolute rate
    """
    return computed_values["traced_flow_rate"]


def get_infectiousness_level(compartment, clinical, non_sympt_infect_multiplier, late_infect_multiplier):
    if clinical == Clinical.NON_SYMPT:
        return non_sympt_infect_multiplier
    elif compartment == Compartment.LATE_ACTIVE and clinical in NOTIFICATION_CLINICAL_STRATA:
        return late_infect_multiplier[clinical]
    else:
        return 1.


@numba.jit(nopython=True)
def get_proportion_detect_force_infection(compartment_values, notif_comps, notif_levels, non_notif_comps, non_notif_levels):
    """
    Calculate the proportion of the force of infection arising from ever-detected individuals
    See PropIndexDetectedProc for details on calling this function
    """
    non_detected_detected_force_of_infection = 0.
    for i, c in enumerate(non_notif_comps):
        non_detected_detected_force_of_infection += compartment_values[c] * non_notif_levels[i]
        
    detected_force_of_infection = 0.
    for i, c in enumerate(notif_comps):
        detected_force_of_infection += compartment_values[c] * notif_levels[i]
        
    total_force_of_infection = non_detected_detected_force_of_infection + detected_force_of_infection

    if total_force_of_infection == 0.:
        return 0.
    else:
        proportion_detect_force_infection = detected_force_of_infection / total_force_of_infection
        assert 0. <= proportion_detect_force_infection <= 1.
        return proportion_detect_force_infection


class PrevalenceProc(ComputedValueProcessor):
    """
    Calculate prevalence from the active disease compartments.
    """
    def __init__(self):
        self.active_comps = None

    def prepare_to_run(self, compartments, flows):
        """
        Identify the compartments with active disease for the prevalence calculation
        """
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)

    def process(self, compartment_values, computed_values, time):
        """
        Calculate the actual prevalence during run-time
        """
        return find_sum(compartment_values[self.active_comps]) / find_sum(compartment_values)


class PropDetectedTracedProc(ComputedValueProcessor):
    """
    Calculate the proportion of detected cases which have their contacts traced.
    """
    def __init__(self, trace_param):
        self.trace_param = trace_param

    def process(self, compartment_values, computed_values, time):
        """
        Formula for calculating the proportion from the already-processed contact tracing parameter,
        which has been worked out in the get_tracing_param function above.
        Ensures that the proportion is bounded [0, 1]
        """
        proportion_of_detected_traced = np.exp(-computed_values["prevalence"] * self.trace_param)
        assert 0. <= proportion_of_detected_traced <= 1.
        return proportion_of_detected_traced


class PropIndexDetectedProc(ComputedValueProcessor):
    """
    Calculate the proportion of all contacts whose index case is ever detected.
    """
    def __init__(self, non_sympt_infect_multiplier, late_infect_multiplier):
        self.non_sympt_infect_multiplier = non_sympt_infect_multiplier
        self.late_infect_multiplier = late_infect_multiplier

    def prepare_to_run(self, compartments, flows):
        """
        Identify the infectious compartments for the prevalence calculation by infection stage and clinical status.
        Also captures the infectiousness levels by infection stage and clinical status.
        """
        notif_comps = []
        non_notif_comps = []
        notif_levels = []
        non_notif_levels = []

        for compartment in INFECTIOUS_COMPARTMENTS:
            for clinical in CLINICAL_STRATA:
                
                infectiousness_level = get_infectiousness_level(
                    compartment, clinical, self.non_sympt_infect_multiplier, self.late_infect_multiplier)
            
                # Get all the matching compartments for the current infectious/stratification level
                cur_comps = [idx for idx, comp in enumerate(compartments) if
                     comp.has_name(compartment) and
                     comp.has_stratum("clinical", clinical)]
                
                # Store these separately as notifying and non-notifying
                if clinical in NOTIFICATION_CLINICAL_STRATA:
                    notif_comps += cur_comps
                    notif_levels += [infectiousness_level] * len(cur_comps)
                else:
                    non_notif_comps += cur_comps
                    non_notif_levels += [infectiousness_level] * len(cur_comps)
                    
        # Convert the indices and levels to numpy arrays
        self.notif_comps = np.array(notif_comps, dtype=int)
        self.notif_levels = np.array(notif_levels, dtype=float)
        self.non_notif_comps = np.array(non_notif_comps, dtype=int)
        self.non_notif_levels = np.array(non_notif_levels, dtype=float)

    def process(self, compartment_values, computed_values, time):
        """
        Calculate the proportion of the force of infection arising from ever-detected individuals
        """
        # Call the optimized numba JIT version of this function (we cannot JIT directly on the class member function)
        return get_proportion_detect_force_infection(compartment_values, self.notif_comps, self.notif_levels, self.non_notif_comps, self.non_notif_levels)


class TracedFlowRateProc(ComputedValueProcessor):
    """
    Calculate the transition flow rate based on the only other outflow and the proportion of all new cases traced.
    """
    def __init__(self, incidence_flow_rate):
        self.incidence_flow_rate = incidence_flow_rate

    def process(self, compartment_values, computed_values, time):
        """
        Solving the following equation:

        traced_prop = traced_flow_rate / (traced_flow_rate + incidence_flow_rate)

        for traced_flow_rate gives the following:
        """
        traced_prop = computed_values["prop_detected_traced"] * computed_values["prop_contacts_with_detected_index"]
        traced_flow_rate = self.incidence_flow_rate * traced_prop / (1. - traced_prop)
        assert 0. <= traced_flow_rate
        return traced_flow_rate
