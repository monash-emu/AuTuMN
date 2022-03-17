from typing import Optional, Dict, List

import pandas as pd
from summer import StrainStratification, Multiply, CompartmentalModel

from autumn.models.sm_sir.constants import Compartment, FlowName
from autumn.models.sm_sir.parameters import VocComponent
from autumn.tools.utils.utils import multiply_function_or_constant


def make_voc_seed_func(entry_rate: float, start_time: float, seed_duration: float):
    """
    Create a simple step function to allow seeding of the VoC strain at a particular point in time.

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
        susc_adjs: pd.Series,
):
    """
    Apply the reinfection flows, making sure that it is possible to be infected with any strain after infection with any
    strain. We'll work out whether this occurs at a reduced rate because of immunity later.

    Args:
        model: The SM-SIR model being adapted
        base_compartments: The unstratified model compartments
        infection_dest: Where people end up first after having been infected
        age_groups: The modelled age groups
        voc_params: The VoC-related parameters
        strain_strata: The strains being implemented or a list of an empty string if no strains in the model
        contact_rate: The model's contact rate
        susc_adjs: Adjustments to the rate of infection of susceptibles based on modelled age groups

    """

    for dest_strain in strain_strata:
        strain_adjuster = voc_params[dest_strain].contact_rate_multiplier
        dest_filter = {"strain": dest_strain}
        for source_strain in strain_strata:
            source_filter = {"strain": source_strain}
            for age_group in age_groups:
                age_filter = {"agegroup": age_group}
                dest_filter.update(age_filter)
                source_filter.update(age_filter)

                contact_rate_adjuster = strain_adjuster * susc_adjs[age_group]
                strain_age_contact_rate = multiply_function_or_constant(contact_rate, contact_rate_adjuster)

                model.add_infection_frequency_flow(
                    FlowName.EARLY_REINFECTION,
                    strain_age_contact_rate,
                    Compartment.RECOVERED,
                    infection_dest,
                    source_filter,
                    dest_filter,
                )
                if Compartment.WANED in base_compartments:
                    model.add_infection_frequency_flow(
                        FlowName.LATE_REINFECTION,
                        strain_age_contact_rate,
                        Compartment.WANED,
                        infection_dest,
                        source_filter,
                        dest_filter,
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
    adjustments = {strain: None for strain in strains}  # Start from a blank adjustments sets
    adjustments_active = {strain: None for strain in strains}
    for strain in strains:
        if voc_params[strain].relative_latency:
            adjustments.update({strain: Multiply(1. / voc_params[strain].relative_latency)})  # Update for user requests
        if voc_params[strain].relative_active_period:
            adjustments_active.update({strain: Multiply(1. / voc_params[strain].relative_active_period)})
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
