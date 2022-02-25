from typing import List, Dict, Optional

from summer import CompartmentalModel

from autumn.models.sm_sir.parameters import VocComponent
from autumn.models.sm_sir.constants import Compartment, FlowName
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
        voc_params: Optional[Dict[str, VocComponent]],
        strain_strata: List[str],
        contact_rate: float,
):
    """
    Apply the reinfection flows, making sure that it is possible to be infected with any strain after infection with any
    strain. We'll work out whether this occurs at a reduced rate because of immunity later.

    Args:
        model: The SM-SIR model being adapted
        base_compartments: The unstratified model compartments
        infection_dest: Where people end up first after having been infected
        voc_params: The VoC-related parameters
        strain_strata: The strains being implemented or a list of an empty string if no strains in the model
        contact_rate: The model's contact rate

    """

    for dest_strain in strain_strata:
        contact_multiplier = voc_params[dest_strain].contact_rate_multiplier
        strain_contact_rate = multiply_function_or_constant(contact_rate, contact_multiplier)
        dest_filter = {"strain": dest_strain}
        for source_strain in strain_strata:
            source_filter = {"strain": source_strain}
            model.add_infection_frequency_flow(
                FlowName.EARLY_REINFECTION,
                strain_contact_rate,
                Compartment.RECOVERED,
                infection_dest,
                source_filter,
                dest_filter,
            )
            if "waned" in base_compartments:
                model.add_infection_frequency_flow(
                    FlowName.LATE_REINFECTION,
                    strain_contact_rate,
                    Compartment.WANED,
                    infection_dest,
                    source_filter,
                    dest_filter,
                )


def apply_reinfection_flows_without_strains(
        model: CompartmentalModel,
        base_compartments: List[str],
        infection_dest: str,
        contact_rate: float,
):
    """
    Apply the reinfection flows, making sure that it is possible to be infected with any strain after infection with any
    strain. We'll work out whether this occurs at a reduced rate because of immunity later.

    Args:
        model: The SM-SIR model being adapted
        base_compartments: The unstratified model compartments
        infection_dest: Where people end up first after having been infected
        contact_rate: The model's contact rate

    """

    model.add_infection_frequency_flow(
        FlowName.EARLY_REINFECTION,
        contact_rate,
        Compartment.RECOVERED,
        infection_dest,
    )
    if "waned" in base_compartments:
        model.add_infection_frequency_flow(
            FlowName.LATE_REINFECTION,
            contact_rate,
            Compartment.WANED,
            infection_dest,
        )
