import numpy as np

from autumn.models.covid_19.constants import Compartment

from summer.compute import DerivedValueProcessor, find_sum


class PrevalenceProc(DerivedValueProcessor):
    def __init__(self, compartments, flows):
        # Anything relating to model structure (indices etc) should be computed in here
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)

    def process(self, comp_vals, flow_rates, time):
        # This is the actual calculation performed at each timestep
        return find_sum(comp_vals[self.active_comps]) / find_sum(comp_vals)


def trace_function(prop_traced, incidence_flow_rate, untraced_comp):
    def contact_tracing_func(
            flow, compartments, compartment_values, flows, flow_rates, derived_values, time
    ):

        #Currently unused, but proof of concept - unfortunately adds ridiculously to run-time
        #active_comps = \
        #   [idx for idx, comp in enumerate(compartments) if
        #    comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)]
        #prevalence = sum(compartment_values[active_comps]) / sum(compartment_values)

        prevalence = derived_values['prevalence']

        return incidence_flow_rate * prop_traced * compartment_values[flow.source.idx]

    return contact_tracing_func
