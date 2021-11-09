from autumn.tools.utils.outputsbuilder import OutputsBuilder


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self):
        self.model.request_output_for_flow(name="incidence", flow_name="infection")

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")
