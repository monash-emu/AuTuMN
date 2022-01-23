from typing import Dict

from summer import CompartmentalModel
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
