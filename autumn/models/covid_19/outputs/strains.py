from summer import CompartmentalModel

from autumn.models.covid_19.constants import INCIDENCE
from autumn.models.covid_19.stratifications.strains import Strain
from autumn.tools.utils.utils import list_element_wise_division


def request_strain_outputs(model: CompartmentalModel, voc_names: list):
    """
    Outputs relating to variants of concern (VoCs)
    """

    # Incidence
    all_strains = [Strain.WILD_TYPE] + voc_names
    for strain in all_strains:
        incidence_key = f"{INCIDENCE}_strain_{strain}"
        model.request_output_for_flow(
            name=incidence_key,
            flow_name=INCIDENCE,
            dest_strata={"strain": strain},
            save_results=False
        )
        # Calculate the proportion of incident cases that are VoC
        model.request_function_output(
            name=f"prop_{INCIDENCE}_strain_{strain}",
            func=lambda strain_inc, total_inc: list_element_wise_division(strain_inc, total_inc),
            sources=[incidence_key, INCIDENCE]
        )
