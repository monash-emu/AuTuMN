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
    Multiply the flow rate through by the source compartment to get the final absolute rate
    """
    return derived_values["traced_flow_rate"] * compartment_values[flow.source.idx]


class PrevalenceProc(DerivedValueProcessor):
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

    def process(self, comp_vals, flow_rates, derived_values, time):
        """
        Calculate the actual prevalence during run-time
        """
        return find_sum(comp_vals[self.active_comps]) / find_sum(comp_vals)


class PropDetectedTracedProc(DerivedValueProcessor):
    """
    Calculate the proportion of detected cases which have their contacts traced.
    """
    def __init__(self, trace_param):
        self.trace_param = trace_param

    def process(self, comp_vals, flow_rates, derived_values, time):
        """
        Formula for calculating the proportion from the already-processed contact tracing parameter,
        which has been worked out in the get_tracing_param function above.
        Ensures that the proportion is bounded [0, 1]
        """
        proportion_of_detected_traced = np.exp(-derived_values["prevalence"] * self.trace_param)
        assert 0. <= proportion_of_detected_traced <= 1.
        return proportion_of_detected_traced


class PropTracedProc(DerivedValueProcessor):
    """
    Calculate the proportion of the contacts of all active cases that are traced.
    """
    def __init__(self, detected_proportion_func, sympt_props):
        self.detected_proportion_func = detected_proportion_func
        self.average_sympt_proportion = np.mean(sympt_props)

    def process(self, comp_vals, flow_rates, derived_values, time):
        """
        Find the approximate total proportion of all cases that are being traced, which assumes approximately equal
        infectiousness of asymptomatic and symptomatic cases over their infectious period.
        """
        return derived_values["prop_detected_traced"] * \
               self.detected_proportion_func(time) * \
               self.average_sympt_proportion


class TracedFlowRateProc(DerivedValueProcessor):
    """
    Calculate the transition flow rate based on the only other outflow and the proportion of all new cases traced.
    """
    def __init__(self, incidence_flow_rate):
        self.incidence_flow_rate = incidence_flow_rate

    def process(self, comp_vals, flow_rates, derived_values, time):
        """
        Solving the following equation:

        traced_prop = traced_flow_rate / (traced_flow_rate + incidence_flow_rate)

        for traced_flow_rate gives the following:
        """
        traced_flow_rate = \
            self.incidence_flow_rate * derived_values["prop_traced"] / (1. - derived_values["prop_traced"])
        assert 0. <= traced_flow_rate
        return traced_flow_rate
