from autumn.model_features.outputs import OutputsBuilder

from summer2.parameters import DerivedOutput


class TbOutputsBuilder(OutputsBuilder):
    """Helps build derived outputs for the TB model"""

    def __init__(self, model, strata) -> None:
        self.model = model
        self.strata = strata


    def request_compartment_output(self, output_name, compartments, save_results=True):
        self.model.request_output_for_compartments(
            output_name, compartments, save_results=save_results
        )
        for gender_stratum in self.strata:
            # For location-specific mortality calculations
            gen_output_name = f"{output_name}Xgender_{gender_stratum}"
            self.model.request_output_for_compartments(
                gen_output_name,
                compartments,
                strata={"gender": gender_stratum},
                save_results=save_results,
            )


    def request_output_func(self, output_name, sources):
        for s in sources:
            self.model.request_function_output(output_name, DerivedOutput(s) / self.model.timestep, save_results=True)
        for gender_stratum in self.strata:
            gen_output_name = f"{output_name}Xgender_{gender_stratum}"
            gen_sources = [f"{s}Xgender_{gender_stratum}" for s in sources]
            for gs in gen_sources:
                self.model.request_function_output(
                        gen_output_name, DerivedOutput(gs) / self.model.timestep, save_results=True
                )

    def request_flow_output(self, output_name, flow_name, save_results=True):
        self.model.request_output_for_flow(output_name, flow_name, save_results=save_results)
        for gender_stratum in self.strata:
            gen_output_name = f"{output_name}Xgender_{gender_stratum}"
            print(gen_output_name)
            self.model.request_output_for_flow(
                gen_output_name,
                flow_name,
                source_strata={"gender": gender_stratum},
                save_results=save_results,
            )

    def request_aggregation_output(self, output_name, sources, save_results=True):
        self.model.request_aggregate_output(output_name, sources, save_results=save_results)
        # for gender_stratum in self.strata:
        #     # For gender-specific mortality calculations
        #     gen_output_name = f"{output_name}Xgender_{gender_stratum}"
        #     gen_sources = [f"{s}X_{gender_stratum}" for s in sources]
        #     self.model.request_aggregate_output(
        #         gen_output_name, gen_sources, save_results=save_results
        #     )

    def request_normalise_flow_output(self, output_name, source, save_results=True):
        self.model.request_function_output(
            output_name, DerivedOutput(source) / self.model.timestep, save_results=save_results
        )


def calculate_percentage(sub_pop_size, total_pop_size):
    return 100 * sub_pop_size / total_pop_size


def calculate_per_hundred_thousand(sub_pop_size, total_pop_size):
    return 1e5 * sub_pop_size / total_pop_size


def calculate_proportion(sub_pop_size, total_pop_size):
    return sub_pop_size / total_pop_size
