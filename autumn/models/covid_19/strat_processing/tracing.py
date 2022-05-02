import numpy as np
import numba

from summer.compute import ComputedValueProcessor, find_sum

from autumn.models.covid_19.constants import (
    Compartment, INFECTIOUS_COMPARTMENTS, Clinical, CLINICAL_STRATA, NOTIFICATION_CLINICAL_STRATA
)


def get_tracing_param(assumed_trace_prop: float, assumed_prev: float, floor: float) -> float:
    """
    Calculate multiplier for the relationship between traced proportion and prevalence for use in the next function.
    This is based on the assumption that there is a predictable relationship between per capita testing numbers and
    case detection rate.
    We further assume an exponential relationship between these two quantities.
    For each model run, we also assume that we know the coordinates of one point on the curve, and given we have one
    unknown parameter, we can solve to get the parameter for the exponent.

    Solving the following equation to get the returned value:
        floor + (1 - floor) exp -(result * assumed_prev) = assumed_trace_prop

    Args:
        assumed_trace_prop: The proportion of contacts effectively traced at the prevalence of interest
        assumed_prev: The prevalence value of interest
        floor: The minimum proportion that would ever be traced (i.e. the asymptote level)

    Returns:
        The parameter value we need for the exponent (result in the equation above)

    """

    return -np.log((assumed_trace_prop - floor) / (1. - floor)) / assumed_prev


def contact_tracing_func(time, computed_values):
    """
    Multiply the flow rate through by the source compartment to get the final absolute rate.

    Args:
        Standard summer functional flow rate arguments

    """

    return computed_values["traced_flow_rate"]


@numba.jit(nopython=True)
def get_proportion_detect_force_infection(
        comp_values: np.ndarray, notif_comps: np.ndarray, notif_levels: np.ndarray, non_notif_comps: np.ndarray,
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

    Attributes:
        active_comps: The indices of the compartments representing active episodes (not including the incubation period)

    """

    def __init__(self):
        self.active_comps = None

    def prepare_to_run(self, compartments, flows):
        """
        Identify the compartments with active disease for the prevalence calculation.

        Args:
            Standard arguments for this method of a summer computed value processor

        """

        active_comps_list = [
            i_comp for i_comp, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)
        ]
        self.active_comps = np.array(active_comps_list, dtype=int)

    def process(self, compartment_values, computed_values, time):
        """
        Now actually calculate the current prevalence during run-time.

        Args:
            Standard arguments for this method of a summer computed value processor

        Returns:
            Prevalence of active disease at this point in time

        """

        return find_sum(compartment_values[self.active_comps]) / find_sum(compartment_values)


class PropIndexDetectedProc(ComputedValueProcessor):
    """
    Calculate the proportion of all contacts whose index case is ever detected.

    Attributes:
        non_sympt_infect_multiplier: Relative infectiousness of asymptomatic episodes
        late_infect_multiplier: Relative infectiousness of persons undergoing quarantine
        notif_comps: Indices of the compartments being notified
        notif_levels: Infectiousness levels of the compartments being notified
        non_notif_comps: Indices of the compartments not being notified
        non_notif_levels: Infectiousness levels of the compartments not being notified

    """

    def __init__(self, non_sympt_infect_multiplier, late_infect_multiplier):
        self.non_sympt_infect_multiplier = non_sympt_infect_multiplier
        self.late_infect_multiplier = late_infect_multiplier
        self.notif_comps = None
        self.notif_levels = None
        self.non_notif_comps = None
        self.non_notif_levels = None

    def prepare_to_run(self, compartments, flows):
        """
        Identify the infectious compartments for the prevalence calculation by infection stage and clinical status.
        Also captures the infectiousness levels by infection stage and clinical status.

        Args:
            Standard arguments for this method of a summer computed value processor

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

        Args:
            Standard arguments for this method of a summer computed value processor

        Returns:
            The proportion of interest

        """

        # Call the optimised numba JIT version of this function (we cannot JIT directly on the class member function)
        return get_proportion_detect_force_infection(
            compartment_values, self.notif_comps, self.notif_levels, self.non_notif_comps, self.non_notif_levels
        )


class PropDetectedTracedProc(ComputedValueProcessor):
    """
    Calculate the proportion of contacts of successfully detected cases which are traced.

    Attributes:
        trace_param: The exponent parameter governing the relationship between testing rates and CDR
        floor: The minimum CDR possible

    """

    def __init__(self, trace_param, floor):
        self.trace_param = trace_param
        self.floor = floor

    def process(self, compartment_values, computed_values, time):
        """
        Formula for calculating the proportion from the already-processed contact tracing parameter, which was
        calculated in the get_tracing_param function above.

        Args:
            Standard arguments for this method of a summer computed value processor

        """

        # Decreasing exponential function of current prevalence descending from one to the floor value
        current_prevalence = computed_values["prevalence"]
        prop_of_detected_traced = self.floor + (1. - self.floor) * np.exp(-current_prevalence * self.trace_param)

        # Should be impossible for this to fail
        msg = f"Proportion of detectable contacts detected not between floor value and 1: {prop_of_detected_traced}"
        assert self.floor <= prop_of_detected_traced <= 1., msg

        return prop_of_detected_traced


class TracedFlowRateProc(ComputedValueProcessor):
    """
    Calculate the transition flow rate based on the only other outflow and the proportion of all new cases traced.

    Attributes:
        incidence_flow_rate: The total flow rate for all incident cases (i.e. reciprocal of late exposed sojourn time)

    """

    def __init__(self, incidence_flow_rate):
        self.incidence_flow_rate = incidence_flow_rate

    def process(self, compartment_values, computed_values, time):
        """
        Solving the following equation:
            traced_prop = traced_flow_rate / (traced_flow_rate + incidence_flow_rate)
        for traced_flow_rate gives the following:

        Args:
            Standard arguments for this method of a summer computed value processor

        """

        # Proportion of all infections that are contact traced
        traced_prop = computed_values["prop_detected_traced"] * computed_values["prop_contacts_with_detected_index"]

        # Applied to adjust the incidence flow rate
        traced_flow_rate = self.incidence_flow_rate * traced_prop / max((1. - traced_prop), 1e-6)

        # Should be impossible for this to fail
        assert 0. <= traced_flow_rate

        return traced_flow_rate
