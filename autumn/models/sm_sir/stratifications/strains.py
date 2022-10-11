from typing import Optional, Dict, List

import pandas as pd
from summer import StrainStratification, Multiply, CompartmentalModel

from autumn.models.sm_sir.constants import Compartment, FlowName
from autumn.models.sm_sir.parameters import VocComponent
from autumn.core.utils.utils import multiply_function_or_constant
from autumn.model_features.strains import broadcast_infection_flows_over_source


def make_voc_seed_func(entry_rate: float, start_time: float, seed_duration: float):
    """
    Create a simple step function to allow seeding of the VoC strain for the period from the requested time point.
    The function starts from zero, steps up to entry_rate from the start_time and then steps back down again to zero after seed_duration has elapsed.

    Args:
        entry_rate: The entry rate
        start_time: The requested time at which seeding should start
        seed_duration: The number of days that the seeding should go for

    Returns:
        The simple step function

    """

    def voc_seed_func(time: float, computed_values):
        return entry_rate if 0. < time - start_time < seed_duration else 0.

    return voc_seed_func


def seed_vocs(model: CompartmentalModel, all_voc_params: Dict[str, VocComponent], seed_compartment: str):
    """
    Use importation flows to seed VoC cases.

    Generally seeding to the infectious compartment, because unlike Covid model, this compartment always present.

    Note that the entry rate will get repeated for each compartment as the requested compartments for entry are
    progressively stratified after this process is applied (but are split over the previous stratifications of the
    compartment to which this is applied, because the split_imports argument is True).

    Args:
        model: The summer model object
        all_voc_params: The VoC-related parameters
        seed_compartment: The compartment that VoCs should be seeded to

    """

    for voc_name, this_voc_params in all_voc_params.items():
        voc_seed_params = this_voc_params.new_voc_seed
        if voc_seed_params:
            voc_seed_func = make_voc_seed_func(
                voc_seed_params.entry_rate,
                voc_seed_params.start_time,
                voc_seed_params.seed_duration
            )
            model.add_importation_flow(
                f"seed_voc_{voc_name}",
                voc_seed_func,
                dest=seed_compartment,
                dest_strata={"strain": voc_name},
                split_imports=True
            )


def apply_reinfection_flows_with_strains(
        model: CompartmentalModel,
        base_compartments: List[str],
        infection_dest: str,
        age_groups: List[str],
        voc_params: Optional[Dict[str, VocComponent]],
        strain_strata: List[str],
        contact_rate: float,
        suscept_adjs: pd.Series,
):
    """
    Apply the reinfection flows, making sure that it is possible to be infected with any strain after infection with any strain.
    We'll work out whether this occurs at a reduced rate because of immunity later, in the various functions of the immunity.py file.

    Args:
        model: The SM-SIR model being adapted
        base_compartments: The unstratified model compartments
        infection_dest: Where people end up first after having been infected
        age_groups: The modelled age groups
        voc_params: The VoC-related parameters
        strain_strata: The strains being implemented or a list of an empty string if no strains in the model
        contact_rate: The model's contact rate
        suscept_adjs: Adjustments to the rate of infection of susceptibles based on modelled age groups

    """

    # Loop over all infecting strains
    for dest_strain in strain_strata:
        dest_filter = {"strain": dest_strain}

        # Adjust for infectiousness of infecting strain
        strain_adjuster = voc_params[dest_strain].contact_rate_multiplier

        # Loop over all age groups
        for age_group in age_groups:
            age_filter = {"agegroup": age_group}
            dest_filter.update(age_filter)
            source_filter = age_filter

            # Get an adjuster that considers both the relative infectiousness of the strain and the relative susceptibility of the age group
            contact_rate_adjuster = strain_adjuster * suscept_adjs[age_group]
            strain_age_contact_rate = multiply_function_or_constant(contact_rate, contact_rate_adjuster)

            # Need to broadcast the flows over the recovered status for the strains
            broadcast_infection_flows_over_source(
                model, 
                FlowName.EARLY_REINFECTION,
                Compartment.RECOVERED,
                infection_dest,
                source_filter, 
                dest_filter,
                strain_age_contact_rate,
                exp_flows=1,
            )
            if Compartment.WANED in base_compartments:
                broadcast_infection_flows_over_source(
                    model,
                    FlowName.LATE_REINFECTION,
                    Compartment.WANED,
                    infection_dest,
                    source_filter,
                    dest_filter,
                    strain_age_contact_rate,
                    exp_flows=1,
                )


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

    # Create the stratification object
    strain_strat = StrainStratification("strain", strains, affected_compartments)

    # Assign the starting population
    population_split = {strain: voc_params[strain].seed_prop for strain in strains}
    strain_strat.set_population_split(population_split)

    # Adjust the contact rate
    transmissibility_adjustment = {strain: Multiply(voc_params[strain].contact_rate_multiplier) for strain in strains}
    strain_strat.set_flow_adjustments(FlowName.INFECTION, transmissibility_adjustment)

    # Start from blank adjustments sets
    adjustments_latent = {strain: None for strain in strains}  
    adjustments_active = {strain: None for strain in strains}    
    for strain in strains:
        if voc_params[strain].relative_latency:
            adjustments_latent.update(
                {strain: Multiply(1. / voc_params[strain].relative_latency)}
            )
        if voc_params[strain].relative_active_period:
            adjustments_active.update(
                {strain: Multiply(1. / voc_params[strain].relative_active_period)}
            )        
    
    # Adjust the latent compartment transitions
    if Compartment.LATENT_LATE in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.WITHIN_LATENT,
            adjustments=adjustments_latent
        )
    if Compartment.LATENT in compartments:
        strain_strat.set_flow_adjustments(
            FlowName.PROGRESSION,
            adjustments=adjustments_latent
        )
    
    # Adjust the active compartment transitions
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
