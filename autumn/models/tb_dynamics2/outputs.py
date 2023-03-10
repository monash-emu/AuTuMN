from autumn.model_features.outputs import OutputsBuilder

from summer2.parameters import DerivedOutput


class TbOutputsBuilder(OutputsBuilder):
    """Helps build derived outputs for the TB model"""

    def __init__(self, model) -> None:
        self.model = model

    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )


    def request_function_output(self, output_name, func, save_results=True):
        self.model.request_function_output(output_name, func, save_results=save_results)

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)

    def request_normalise_flow_output(self, output_name, source, save_results=True):
        self.model.request_function_output(
            output_name, DerivedOutput(source) / self.model.timestep, save_results=save_results
        )

