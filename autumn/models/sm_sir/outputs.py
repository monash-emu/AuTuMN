from autumn.tools.utils.outputsbuilder import OutputsBuilder
from .constants import AGEGROUP_STRATA, FlowName


class SmSirOutputsBuilder(OutputsBuilder):

    def request_incidence(self):
        self.model.request_output_for_flow(name="incidence", flow_name=FlowName.INFECTION)

        # Stratified by age group
        self.request_stratified_output_for_flow(FlowName.INFECTION, AGEGROUP_STRATA, "agegroup", name_stem="incidence")

    def request_mortality(self):
        self.model.request_output_for_flow(name="infection_death", flow_name=FlowName.INFECTION_DEATH)

        # Stratified by age group
        self.request_stratified_output_for_flow(FlowName.INFECTION_DEATH, AGEGROUP_STRATA, "agegroup", filter_on="source")

    def request_hospitalisations(self, prop_symptomatic, prop_hospital):
        hospital_sources = []
        # request age-stratified outputs
        for i, agegroup in enumerate(AGEGROUP_STRATA):
            output_name = f"hospital_admissionsXagegroup_{agegroup}"
            hospital_sources.append(output_name)
            prop_symp = prop_symptomatic[i]
            prop_hosp = prop_hospital[i]
            self.model.request_function_output(
                name=output_name,
                func=make_hospitalisations_func(prop_symp, prop_hosp),
                sources=[f"incidenceXagegroup_{agegroup}"],
            )
        # request unstratified output
        self.model.request_aggregate_output(name="hospital_admissions", sources=hospital_sources)

    def request_random_process_outputs(self,):
        self.model.request_computed_value_output("transformed_random_process")


def make_hospitalisations_func(prop_symp, prop_hosp):

    def calculate_hospitalisations(incidence):
        return incidence * prop_symp * prop_hosp
    
    return calculate_hospitalisations
