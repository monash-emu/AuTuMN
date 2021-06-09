import numpy as np

from autumn.models.covid_19.constants import Compartment
from autumn.models.covid_19.preprocess.case_detection import build_detected_proportion_func

from summer.compute import DerivedValueProcessor, find_sum


class TracingProc(DerivedValueProcessor):
    def __init__(self, agegroup_strata, country, pop, testing_to_detection, case_detection):
        # Initialise this with any additional parameters or data required (add arguments as necessary)
        self.agegroup_strata = agegroup_strata
        self.country = country
        self.pop = pop
        self.testing_to_detection = testing_to_detection
        self.case_detection = case_detection

    def prepare_to_run(self, compartments, flows):
        # Anything relating to model structure (indices etc) should be computed in here
        self.active_comps = np.array([idx for idx, comp in enumerate(compartments) if
            comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)], dtype=int)
        self.get_detected_proportion = build_detected_proportion_func(
            self.agegroup_strata, self.country, self.pop, self.testing_to_detection, self.case_detection
        )

    def process(self, comp_vals, flow_rates, time):
        # This is the actual calculation performed at each timestep
        print(self.get_detected_proportion(time))
        return


def trace_function(prop_traced, incidence_flow_rate):
    def contact_tracing_func(
            flow, compartments, compartment_values, flows, flow_rates, derived_values, time
    ):
        return incidence_flow_rate * prop_traced * compartment_values[flow.source.idx]

    return contact_tracing_func
