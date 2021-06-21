import numpy as np

from autumn.models.covid_19.constants import Compartment

from summer.compute import DerivedValueProcessor, find_sum


def get_tracing_param(assumed_trace_prop, assumed_prev):
    """
    Calculate multiplier for the relationship between traced proportion and prevalence for use in the next function.
    """
    assert 0. <= assumed_trace_prop <= 1.
    assert 0. <= assumed_prev <= 1.
    return -np.log(assumed_trace_prop) / assumed_prev


def get_traced_prop(trace_param, prev):
    """
    Function for the proportion of detected people who are traced.
    """
    return np.exp(-prev * trace_param)


def contact_tracing_func(flow, compartments, compartment_values, flows, flow_rates, derived_values, time):
    """
    Calculate the transition flow as the product of the size of the source compartment, the only outflow and the
    proportion of all new cases traced.

    Solving the following equation:

    traced_prop = traced_flow_rate / (traced_flow_rate + incidence_flow_rate)

    for traced_flow_rate gives the following:
    """
    #traced_flow_rate = incidence_flow_rate * derived_values["prop_traced"] / (1. - derived_values["prop_traced"])

    # Multiply through by the source compartment to get the final absolute rate
    return derived_values['traced_flow_rate'] * compartment_values[flow.source.idx]


class PrevalenceProc(DerivedValueProcessor):
    """
        Calculate prevalence from active disease compartments
    """

    def prepare_to_run(self, compartments, flows):
        # Identify the compartments with active disease for the prevalence calculation
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)

    def process(self, comp_vals, flow_rates, derived_values, time):
        prevalence = find_sum(comp_vals[self.active_comps]) / find_sum(comp_vals)  # Calculate prevalence
        return prevalence

class PropDetectedTracedProc(DerivedValueProcessor):
    """
        Calculate the proportion of detected cases which are traced
    """
    def __init__(self, trace_param):
        self.trace_param = trace_param

    def process(self, comp_vals, flow_rates, derived_values, time):
        prop_detected_traced = get_traced_prop(self.trace_param, derived_values['prevalence'])  # Find the prop of detected that is traced
        return prop_detected_traced


class PropTracedProc(DerivedValueProcessor):
    """
        Calculate the proportion of all cases traced
    """
    def __init__(self, detected_prop_func):
        self.detected_prop_func = detected_prop_func

    def process(self, comp_vals, flow_rates, derived_values, time):
        prop_traced = derived_values['prop_detected_traced'] * self.detected_prop_func(time)  # Last, find the prop of all cases traced
        return prop_traced

class TracedFlowRateProc(DerivedValueProcessor):
    def __init__(self, incidence_flow_rate):
        self.incidence_flow_rate = incidence_flow_rate

    def process(self, comp_vals, flow_rates, derived_values, time):
        traced_flow_rate = self.incidence_flow_rate * derived_values["prop_traced"] / (1. - derived_values["prop_traced"])
        return traced_flow_rate

class TracingProc(DerivedValueProcessor):
    """
        Calculate the proportion of traced cases
        +++ FIXME This is redundant now, but has been kept in for verification puruposes
    """
    def __init__(self, trace_param, detected_prop_func):
        """
        Arguments needed to calculate running quantities during run-time.
        """
        self.trace_param = trace_param
        self.detected_prop_func = detected_prop_func
        self.active_comps = None
        self.get_traced_prop = None

        # Get the function for the proportion of contacts of detected cases who are traced
        self.get_traced_prop = get_traced_prop

    def prepare_to_run(self, compartments, flows):
        """
        Calculate emergent quantities from the model during run-time
        To avoid calculating these quantities for every compartment in function flows
        """

        # Identify the compartments with active disease for the prevalence calculation
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)

    def process(self, comp_vals, flow_rates, derived_values, time):
        """
        The actual calculation performed during run-time
        Calculate the actual proportion of detected cases detected
        """
        prevalence = find_sum(comp_vals[self.active_comps]) / find_sum(comp_vals)  # Calculate prevalence
        prop_detected_traced = self.get_traced_prop(self.trace_param, prevalence)  # Find the prop of detected that is traced
        prop_traced = prop_detected_traced * self.detected_prop_func(time)  # Last, find the prop of all cases traced

        return prop_traced
