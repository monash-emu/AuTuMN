from typing import Dict, List, Union

from summer import CompartmentalModel

from autumn.models.sm_sir.constants import Compartment, FlowName
from autumn.models.sm_sir.parameters import VocComponent


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


def seed_vocs(model: CompartmentalModel, voc_params: Dict[str, VocComponent], seed_compartment: str):
    """
    Use importation flows to seed VoC cases.

    Generally seeding to the infectious compartment, because unlike Covid model, this compartment always present.

    Note that the entry rate will get repeated for each compartment as the requested compartments for entry are
    progressively stratified after this process is applied (but are split over the previous stratifications of the
    compartment to which this is applied, because the split_imports argument is True).

    Args:
        model: The summer model object
        voc_params: The VoC-related parameters
        seed_compartment: The compartment that VoCs should be seeded to

    """

    for voc_name, voc_values in voc_params.items():
        voc_seed_func = make_voc_seed_func(
            voc_values.entry_rate,
            voc_values.start_time,
            voc_values.seed_duration
        )
        model.add_importation_flow(
            f"seed_voc_{voc_name}",
            voc_seed_func,
            dest=seed_compartment,
            dest_strata={"strain": voc_name},
            split_imports=True
        )


def modify_function_or_value(function_or_value, modification):
    if callable(function_or_value):
        def modified_contact_rate(t, c):
            return function_or_value(t, c) * modification
        return modified_contact_rate
    else:
        return function_or_value * modification


def add_strain_cross_protection(
        model: CompartmentalModel,
        base_compartments: List[str],
        infection_dest: str,
        contact_rate: Union[float, callable],
        strain_strata: List[str],
        cross_protection_params: dict
):
    """
    Apply infection flows for the early and late recovered compartments, accounting for cross immunity between strains.

    Args:
        model: The summer model object to be modified
        base_compartments: The base compartment names
        infection_dest: The name of the compartment that people enter as they are infected
        contact_rate: The contact rate, which may be constant or time-varying
        strain_strata: The names of the strains being implemented
        cross_protection_params: The parameters that govern cross-infection

    """

    # Considering recovery with one particular modelled strain ...
    for infected_strain, infecting_strains in cross_protection_params.items():
        infected_strain_params = cross_protection_params[infected_strain]
        msg = "Strain cross immunity incorrectly specified"
        assert list(infected_strain_params.keys()) == strain_strata, msg

        # ... and its protection against infection with a new index strain.
        for infecting_strain in infecting_strains:
            strain_combination_protections = cross_protection_params[infected_strain][infecting_strain]
            expected_flows_list = [FlowName.EARLY_REINFECTION, FlowName.LATE_REINFECTION]
            msg = "Flows to which strain cross immunity applied incorrectly specified"
            assert list(strain_combination_protections.keys()) == expected_flows_list, msg

            # Apply the modification to the early recovered compartment
            modification = 1. - strain_combination_protections[FlowName.EARLY_REINFECTION]
            model.add_infection_frequency_flow(
                name=FlowName.EARLY_REINFECTION,
                contact_rate=modify_function_or_value(contact_rate, modification),
                source=Compartment.RECOVERED,
                dest=infection_dest,
                source_strata={"strain": infected_strain},
                dest_strata={"strain": infecting_strain},
            )

            # Apply the immunity-specific protection to the late recovered or "waned" compartment
            modification = 1. - strain_combination_protections[FlowName.LATE_REINFECTION]
            if "waned" in base_compartments:
                model.add_infection_frequency_flow(
                    name=FlowName.LATE_REINFECTION,
                    contact_rate=modify_function_or_value(contact_rate, modification),
                    source=Compartment.WANED,
                    dest=infection_dest,
                    source_strata={"strain": infected_strain},
                    dest_strata={"strain": infecting_strain},
                )
