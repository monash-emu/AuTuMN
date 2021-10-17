
class Outputs:
    """
    This class is not specific to Covid, so should be moved out of this file - but not sure whether to move it to
    somewhere in autumn or in summer.
    """

    def __init__(self, model, COMPARTMENTS):
        self.model = model
        self.model.request_output_for_compartments(
            name="_total_population", compartments=COMPARTMENTS, save_results=False
        )

    def request_stratified_output_for_flow(self, flow, strata, stratification, name_stem=None, filter_on="destination"):
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
                raise ValueError(f"filter_on should be either 'source' or 'destination': {filter_on}")

    def request_double_stratified_output_for_flow(
            self, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None, filter_on="destination"
    ):
        """
        As for previous function, but looping over two stratifications.
        """

        stem = name_stem if name_stem else flow
        for stratum_1 in strata_1:
            for stratum_2 in strata_2:
                name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
                if filter_on == "destination":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        dest_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                elif filter_on == "source":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        source_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                else:
                    raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")

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
