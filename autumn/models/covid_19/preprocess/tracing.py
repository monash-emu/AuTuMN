import numpy as np

from autumn.models.covid_19.constants import Compartment
from autumn.models.covid_19.preprocess.case_detection import build_detected_proportion_func

from summer.compute import DerivedValueProcessor, find_sum


def get_traced_prop_factory(trace_param):
    """
    Create function for the proportion of detected people who are traced.
    """
    def get_traced_prop(prev):
        return np.exp(-prev * trace_param)
    return get_traced_prop


def trace_function(incidence_flow_rate):
    def contact_tracing_func(
            flow, compartments, compartment_values, flows, flow_rates, derived_values, time
    ):
        """
        Calculate the transition flow as the product of the size of the source compartment, the only outflow and the
        proportion of all new cases traced.

        Solving the following equation:

        traced_prop = traced_flow_rate / (traced_flow_rate + incidence_flow_rate)

        for traced_flow_rate gives the following equation:
        """
        traced_flow_rate = incidence_flow_rate * derived_values["traced_prop"] / (1. - derived_values["traced_prop"])

        # Multiply through by the source compartment to get the final absolute rate
        return traced_flow_rate * compartment_values[flow.source.idx]

    return contact_tracing_func


class TracingProc(DerivedValueProcessor):
    def __init__(self, trace_param, agegroup_strata, country, pop, testing_to_detection, case_detection):
        """
        Arguments needed to calculate running quantities during run-time.
        """
        self.trace_param = trace_param
        self.agegroup_strata = agegroup_strata
        self.country = country
        self.pop = pop
        self.testing_to_detection = testing_to_detection
        self.case_detection = case_detection
        self.active_comps = None
        self.get_detected_proportion = None
        self.get_traced_prop = None

    def prepare_to_run(self, compartments, flows):
        """
        Calculate emergent quantities from the model during run-time
        To avoid calculating these quantities for every compartment in function flows
        """

        # Identify the compartments with active disease for the prevalence calculation
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)

        # Create the CDR function in exactly the same way as what is used in calculating the flow rates
        self.get_detected_proportion = build_detected_proportion_func(
            self.agegroup_strata, self.country, self.pop, self.testing_to_detection, self.case_detection
        )

        # Get the function for the proportion of contacts of detected cases who are traced
        self.get_traced_prop = get_traced_prop_factory(self.trace_param)

    def process(self, comp_vals, flow_rates, time):
        """
        The actual calculation performed during run-time
        Calculate the actual proportion of detected cases detected
        """
        prev = find_sum(comp_vals[self.active_comps]) / find_sum(comp_vals)
        prop_detected_traced = self.get_traced_prop(prev)
        prop_traced = prop_detected_traced * self.get_detected_proportion(time)
        return prop_traced
