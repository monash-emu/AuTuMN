from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import AGEGROUP_STRATA, IMMUNITY_STRATA,FlowName
import numpy as np


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self):
        self.model.request_output_for_flow(name="incidence", flow_name=FlowName.INFECTION)

        # Stratified by age group and immunity status
        self.request_double_stratified_output_for_flow(
            FlowName.INFECTION,
            AGEGROUP_STRATA,
            "agegroup",
            IMMUNITY_STRATA,
            "immunity",
            name_stem="incidence"
        )

    def request_hospital_occupancies(self):

        self.model.request_function_output(
            name="hospital_occupancy",
            sources=["incidenceXagegroup_15Ximmunity_low"],
            func=calc_hospital_occupancy)

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")


def calc_hospital_occupancy(inc):

    out = np.zeros_like(inc)

    for i in range(inc.size):
        if i > 4:
            out[i] = inc[i - 5]

    return out
