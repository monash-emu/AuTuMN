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
    strains = list(voc_params.keys())
    affected_compartments = [comp for comp in compartments if comp != Compartment.SUSCEPTIBLE]

    # Check only one strain is specified as the starting strain
    msg = "More than one strain has been specified as the starting strain"
    assert [voc_params[i_strain].starting_strain for i_strain in strains].count(True) == 1, msg
    starting_strain = [i_strain for i_strain in strains if voc_params[i_strain].starting_strain][0]

    # Create the stratification object
    strain_strat = StrainStratification("strain", strains, affected_compartments)

    # Population split
    msg = "Strain seed proportions do not sum to one"
    assert sum([voc_params[i_strain].seed_prop for i_strain in strains]) == 1., msg
    msg = "Currently requiring starting seed to all be assigned to the strain nominated as the starting strain"
    assert voc_params[starting_strain].seed_prop == 1., msg
    population_split = {strain: voc_params[strain].seed_prop for strain in strains}
    strain_strat.set_population_split(population_split)

    # Transmissibility adjustment
    msg = "Starting strain should have a null contact rate multiplier"
    assert voc_params[starting_strain].contact_rate_multiplier is None, msg
    transmissibility_adjustments = {strain: Multiply(voc_params[strain].contact_rate_multiplier) for strain in strains}
    strain_strat.set_flow_adjustments(FlowName.INFECTION, transmissibility_adjustments)

    return strain_strat
