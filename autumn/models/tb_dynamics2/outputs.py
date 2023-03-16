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
        # for gender_stratum in self.gender_strata:
        #     # For location-specific mortality calculations
        #     gender_output_name = f"{output_name}Xgender_{gender_stratum}"
        #     self.model.request_output_for_compartments(
        #         gender_output_name,
        #         compartments,
        #         strata={"gender": gender_stratum},
        #         save_results=save_results,
        #     )


    # def request_function_output_notifs(self, output_name):
    #     self.model.request_function_output(output_name, DerivedOutput("passive_notifications_raw") / self.model.timestep, save_results=True)
    #     for gender_stratum in self.gender_strata:
    #         gender_output_name = f"{output_name}Xgender_{gender_stratum}"
    #         source = f"passive_notifications_rawXgender_{gender_stratum}" 
    #         self.model.request_function_output(
    #             gender_output_name, DerivedOutput(source) / self.model.timestep , save_results=True
    #         )

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)
        # for gender_stratum in self.gender_strata:
        #     gender_output_name = f"{output_name}Xgender_{gender_stratum}"
        #     self.model.request_output_for_flow(
        #         gender_output_name,
        #         flow_name,
        #         source_strata={"gender": gender_stratum},
        #         save_results=save_results,
        #     )

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)

    def request_normalise_flow_output(self, output_name, source, save_results=True):
        self.model.request_function_output(
            output_name, DerivedOutput(source) / self.model.timestep, save_results=save_results
        )

