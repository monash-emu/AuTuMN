from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import AGEGROUP_STRATA, IMMUNITY_STRATA,FlowName


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

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")
