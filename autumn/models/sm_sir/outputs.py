from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import AGEGROUP_STRATA


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self):
        self.model.request_output_for_flow(name="incidence", flow_name="infection")

        # Stratified by age group
        self.request_stratified_output_for_flow("infection", AGEGROUP_STRATA, "agegroup", name_stem="incidence")

    def request_hospitalisations(self, prop_symptomatic, prop_hospital):
        hospital_sources = []
        # request age-stratified outputs
        for i, agegroup in enumerate(AGEGROUP_STRATA):
            output_name = f"hospital_admissionsXagegroup_{agegroup}"
            hospital_sources.append(output_name)
            self.model.request_function_output(
                name=output_name,
                func=lambda incidence: incidence * prop_symptomatic[i] * prop_hospital[i],
                sources=[f"incidenceXagegroup_{agegroup}"],
            )
        # request unstratified output
        self.model.request_aggregate_output(name="hospital_admissions", sources=hospital_sources)

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")
