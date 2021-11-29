from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import AGEGROUP_STRATA


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self):
        self.model.request_output_for_flow(name="incidence", flow_name="infection")

        # Stratified by age group
        self.request_stratified_output_for_flow("infection", AGEGROUP_STRATA, "agegroup", name_stem="incidence")

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")
