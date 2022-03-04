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
        compartments: All the model's unstratified compartment types

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

    # Latency progression rate adjustment - applies to zero, one or two flows relating to progression through latency
    adjustments = {strain: None for strain in strains}  # Start from a blank adjustments set
    adjustments_active = {strain: None for strain in strains}  # Start from a blank adjustments set
    for strain in strains:
        if voc_params[strain].relative_latency:
            adjustments.update({strain: Multiply(1. / voc_params[strain].relative_latency)})  # Update for user request
        if voc_params[strain].relative_active_period:
            adjustments_active.update({strain: Multiply(1. / voc_params[strain].relative_active_period)})  # Update for user request
    if Compartment.LATENT_LATE in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.WITHIN_LATENT,
            adjustments=adjustments
        )
    if Compartment.LATENT in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.PROGRESSION,
            adjustments=adjustments
        )

    if Compartment.INFECTIOUS_LATE in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.WITHIN_INFECTIOUS,
            adjustments=adjustments_active
        )
    if Compartment.INFECTIOUS in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.RECOVERY,
            adjustments=adjustments_active
        )

    return strain_strat
