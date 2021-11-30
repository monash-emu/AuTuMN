import numpy as np
import numba

from summer.compute import ComputedValueProcessor, find_sum

from autumn.models.covid_19.constants import (
    Compartment, INFECTIOUS_COMPARTMENTS, Clinical, CLINICAL_STRATA, NOTIFICATION_CLINICAL_STRATA
)


def get_tracing_param(assumed_trace_prop: float, assumed_prev: float, floor: float) -> float:
    """
    Calculate multiplier for the relationship between traced proportion and prevalence for use in the next function.

    Solving the following equation to get the returned value:
    floor + (1 - floor) exp -(result * assumed_prev) = assumed_trace_prop
    """

    return -np.log((assumed_trace_prop - floor) / (1. - floor)) / assumed_prev


def contact_tracing_func(time, computed_values):
    """
    Multiply the flow rate through by the source compartment to get the final absolute rate.
    """

    return computed_values["traced_flow_rate"]


@numba.jit(nopython=True)
def get_proportion_detect_force_infection(
        comp_values: np.ndarray,
        notif_comps: np.ndarray,
        notif_levels: np.ndarray,
        non_notif_comps: np.ndarray,
        non_notif_levels: np.ndarray
) -> float:
    """
    Calculate the proportion of the force of infection that is attributable to ever-detected individuals.
    See PropIndexDetectedProc for details on calling this function.

    Args:
        comp_values: Model compartment values
        notif_comps: Notified compartments
        notif_levels: Notified compartment values infectiousness levels
        non_notif_comps: Non-notified compartments
        non_notif_levels: Non-notified compartment values infectiousness levels

    Returns:
        Proportion of the force of infection arising from those who have been detected (or zero if FoI is zero)

    """

    detected_foi = 0.
    for i_comp, comp in enumerate(notif_comps):
        detected_foi += comp_values[comp] * notif_levels[i_comp]

    undetected_foi = 0.
    for i_comp, comp in enumerate(non_notif_comps):
        undetected_foi += comp_values[comp] * non_notif_levels[i_comp]

    total_force_of_infection = detected_foi + undetected_foi

    # Return zero if force of infection is zero to prevent division by zero error
    if total_force_of_infection == 0.:
        return 0.

    # Otherwise return the calculated proportion
    else:
        proportion_detect_force_infect = detected_foi / total_force_of_infection

        # Should be impossible to fail this assertion
        msg = "Force of infection not in range [0, 1]"
        assert 0. <= proportion_detect_force_infect <= 1., msg

        return proportion_detect_force_infect


class PrevalenceProc(ComputedValueProcessor):
    """
    Track the current prevalence of all the active disease compartments for later use to determine the efficiency of
    contact tracing.
    """

    def __init__(self):
        self.active_comps = None

    def prepare_to_run(self, compartments, flows):
        """
        Identify the compartments with active disease for the prevalence calculation.

        Args:
            compartments: All the compartments in the model
            flows: All the flows in the model

        """

        active_comps_list = [
            idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)
        ]
        self.active_comps = np.array(active_comps_list, dtype=int)

    def process(self, compartment_values, computed_values, time):
        """
        Now actually calculate the current prevalence during run-time.

        Args:
            compartment_values: All the model compartment values
            computed_values: All the computed values being tracked during run-time
            time: Current time in integration

        Returns:
            Prevalence of active disease at this point in time

        """

        return find_sum(compartment_values[self.active_comps]) / find_sum(compartment_values)


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

        notif_comps, non_notif_comps, notif_levels, non_notif_levels = [], [], [], []

        for compartment in INFECTIOUS_COMPARTMENTS:
            for clinical in CLINICAL_STRATA:

                if clinical == Clinical.NON_SYMPT:
                    infectiousness_level = self.non_sympt_infect_multiplier
                elif compartment == Compartment.LATE_ACTIVE and clinical in NOTIFICATION_CLINICAL_STRATA:
                    infectiousness_level = self.late_infect_multiplier[clinical]
                else:
                    infectiousness_level = 1.

                # Get all the matching compartments for the current infectious/stratification level
                working_comps = [
                    idx for idx, comp in enumerate(compartments) if
                    comp.has_name(compartment) and comp.has_stratum("clinical", clinical)
                ]

                # Store these separately as notified and non-notified
                if clinical in NOTIFICATION_CLINICAL_STRATA:
                    notif_comps += working_comps
                    notif_levels += [infectiousness_level] * len(working_comps)
                else:
                    non_notif_comps += working_comps
                    non_notif_levels += [infectiousness_level] * len(working_comps)

        # Convert the indices and levels to numpy arrays
        self.notif_comps = np.array(notif_comps, dtype=int)
        self.notif_levels = np.array(notif_levels, dtype=float)
        self.non_notif_comps = np.array(non_notif_comps, dtype=int)
        self.non_notif_levels = np.array(non_notif_levels, dtype=float)

    def process(self, compartment_values, computed_values, time):
        """
        Calculate the proportion of the force of infection arising from ever-detected individuals
        """

        # Call the optimised numba JIT version of this function (we cannot JIT directly on the class member function)
        return get_proportion_detect_force_infection(
            compartment_values, self.notif_comps, self.notif_levels, self.non_notif_comps, self.non_notif_levels
        )


class PropDetectedTracedProc(ComputedValueProcessor):
    """
    Calculate the proportion of contacts of successfully detected cases which are traced.
    """

    def __init__(self, trace_param, floor):
        self.trace_param = trace_param
        self.floor = floor

    def process(self, compartment_values, computed_values, time):
        """
        Formula for calculating the proportion from the already-processed contact tracing parameter,
        which has been worked out in the get_tracing_param function above.
        """

        # Decreasing exponential function of current prevalence descending from one to the floor value
        current_prevalence = computed_values["prevalence"]
        prop_of_detected_traced = self.floor + (1. - self.floor) * np.exp(-current_prevalence * self.trace_param)

        msg = f"Proportion of detectable contacts detected not between floor value and 1: {prop_of_detected_traced}"
        assert self.floor <= prop_of_detected_traced <= 1., msg

        return prop_of_detected_traced


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

        # Proportion of all infections that are contact traced
        traced_prop = computed_values["prop_detected_traced"] * computed_values["prop_contacts_with_detected_index"]

        # Applied to adjust the incidence flow rate
        traced_flow_rate = self.incidence_flow_rate * traced_prop / max((1. - traced_prop), 1e-6)
        assert 0. <= traced_flow_rate
        return traced_flow_rate
