from summer import CompartmentalModel, Compartment

from autumn.models.covid_19.parameters import Parameters
from autumn.settings import Region
from autumn.models.covid_19.constants import INFECT_DEATH, INFECTION
from .standard import request_stratified_output_for_flow, request_double_stratified_output_for_flow
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA


def request_common_outputs(model: CompartmentalModel, params: Parameters, is_region_vic):

    """
    Infection
    """

    # susceptible_infection_rate functions only work for SEIR structure, would need to change for SEIRS, SEIS, etc.
    model.request_output_for_flow("infection", INFECTION)
    model.request_output_for_compartments("_susceptible", [Compartment.SUSCEPTIBLE], save_results=False)
    model.request_function_output(
        "susceptible_infection_rate",
        func=lambda infection, susceptible: infection / susceptible,
        sources=[INFECTION, "_susceptible"]
    )

    """
    Deaths
    """

    # Unstratified
    model.request_output_for_flow(name="infection_deaths", flow_name=INFECT_DEATH)
    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

    # Stratified by age
    request_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem="infection_deaths", filter_on="source"
    )
    for agegroup in AGEGROUP_STRATA:
        model.request_cumulative_output(
            name=f"accum_deathsXagegroup_{agegroup}",
            source=f"infection_deathsXagegroup_{agegroup}",
        )

    # Stratified by age and clinical stratum
    request_double_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup",
        CLINICAL_STRATA, "clinical", name_stem="infection_deaths", filter_on="source"
    )

    # Victoria-specific output by cluster
    if is_region_vic:
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
        for cluster in clusters:
            model.request_output_for_flow(
                name=f"infection_deaths_for_cluster_{cluster}",
                flow_name=INFECT_DEATH,
                source_strata={"cluster": cluster},
            )
