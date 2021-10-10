from summer import CompartmentalModel

from autumn.models.covid_19.constants import INCIDENCE
from autumn.models.covid_19.stratifications.strains import Strain
from autumn.tools.utils.utils import list_element_wise_division
from autumn.models.covid_19.outputs.common import request_stratified_output_for_flow


def request_strain_outputs(model: CompartmentalModel, voc_names: list):
    """
    Outputs relating to variants of concern (VoCs).
    """

    # Incidence rate for each strain implemented
    all_strains = [Strain.WILD_TYPE] + voc_names
    request_stratified_output_for_flow(model, INCIDENCE, all_strains, "strain")

    # Convert to a proportion
    for strain in all_strains:
        model.request_function_output(
            name=f"prop_{INCIDENCE}_strain_{strain}",
            func=lambda strain_inc, total_inc: list_element_wise_division(strain_inc, total_inc),
            sources=[f"{INCIDENCE}Xstrain_{strain}", INCIDENCE]
        )
