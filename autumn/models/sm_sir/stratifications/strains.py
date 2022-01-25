from typing import Optional, Dict, List

from summer import StrainStratification, Multiply

from autumn.models.sm_sir.parameters import VocComponent
from autumn.models.sm_sir.constants import Compartment, WILD_TYPE, FlowName


def get_strain_strat(voc_params: Optional[Dict[str, VocComponent]], compartments: List[str]):
    """
    Stratify the model by strain, with at least two strata, being wild or "ancestral" virus type and the variants of
    concern ("VoC").

    We are now stratifying all the compartments, including the recovered ones. The recovered compartment stratified by
    strain represents people whose last infection was with that strain.

    Args:
        voc_params: All the VoC parameters (one VocComponent parameters object for each VoC)
        compartments: All the model's base compartment types

    Returns:
        The strain stratification summer object

    """

    # Process the requests
    voc_names = list(voc_params.keys())
    all_strains = [WILD_TYPE] + voc_names
    affected_compartments = [comp for comp in compartments if comp != Compartment.SUSCEPTIBLE]

    # Stratify model
    strain_strat = StrainStratification("strain", all_strains, affected_compartments)

    # Prepare population split and transmission adjustments
    population_split = {WILD_TYPE: 1.}
    transmissibility_adjustment = {WILD_TYPE: None}
    for voc_name in voc_names:
        population_split[voc_name] = 0.
        transmissibility_adjustment[voc_name] = Multiply(voc_params[voc_name].contact_rate_multiplier)

    # Apply population split
    strain_strat.set_population_split(population_split)

    # Apply transmissibility adjustments
    strain_strat.set_flow_adjustments(FlowName.INFECTION, transmissibility_adjustment)

    return strain_strat
