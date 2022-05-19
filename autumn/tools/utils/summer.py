from summer.compute import ComputedValueProcessor


class FunctionWrapper(ComputedValueProcessor):
    """
    Very basic processor that wraps a time/computed values function
    of the type used in flow and adjusters

    FIXME:
    This is such a basic use case, it probably belongs in summer

    """

    def __init__(self, function_to_wrap: callable):
        """
        Initialise with just the param function
        Args:
            function_to_wrap: The function
        """

        self.wrapped_function = function_to_wrap

    def process(self, compartment_values, computed_values, time):
        return self.wrapped_function(time, computed_values)


class OutputsBuilder:

    def __init__(self, model, compartments):
        self.model = model
        self.compartments = compartments
        self.model.request_output_for_compartments(
            name="total_population", compartments=compartments
        )

    def request_stratified_output_for_flow(
        self, flow, strata, stratification, name_stem=None, filter_on="destination"
    ):
        """
        Standardise looping over stratum to pull out stratified outputs for flow.
        """

        stem = name_stem if name_stem else flow
        for stratum in strata:
            if filter_on == "destination":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    dest_strata={stratification: stratum},
                )
            elif filter_on == "source":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    source_strata={stratification: stratum},
                )
            else:
                raise ValueError(
                    f"filter_on should be either 'source' or 'destination': {filter_on}"
                )

    def request_double_stratified_output_for_flow(
        self,
        flow,
        strata1,
        stratification1,
        strata2,
        stratification2,
        name_stem=None,
        filter_on="destination",
    ):
        """
        As for previous function, but looping over two stratifications.
        """

        stem = name_stem if name_stem else flow
        for stratum_1 in strata1:
            for stratum_2 in strata2:
                name = f"{stem}X{stratification1}_{stratum_1}X{stratification2}_{stratum_2}"
                if filter_on == "destination":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        dest_strata={
                            stratification1: stratum_1,
                            stratification2: stratum_2,
                        },
                    )
                elif filter_on == "source":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        source_strata={
                            stratification1: stratum_1,
                            stratification2: stratum_2,
                        },
                    )
                else:
                    raise ValueError(
                        f"filter_on should be either 'source' or 'destination', found {filter_on}"
                    )

    def request_stratified_output_for_compartment(
        self, request_name, compartments, strata, stratification, save_results=True
    ):
        for stratum in strata:
            full_request_name = f"{request_name}X{stratification}_{stratum}"
            self.model.request_output_for_compartments(
                name=full_request_name,
                compartments=compartments,
                strata={stratification: stratum},
                save_results=save_results,
            )
