from typing import Dict

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, Vaccination
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.vaccination import apply_immunity_to_strat


def get_vaccination_strat(
        params: Parameters, all_strata: list, stratified_adjusters: Dict[str, Dict[str, float]]
) -> Stratification:
    """
    Get the vaccination stratification and adjustments to apply to the model, calling the required functions in the
    vaccination strat_processing folder to organise the parameter processing and inter-compartmental flows representing
    vaccination.

    Args:
        params: All the model parameters
        all_strata: All the vaccination strata being implemented in the model (including unvaccinated)
        stratified_adjusters: VoC and severity stratification adjusters

    Returns:
        The processing summer vaccination stratification object

    """

    # Create the stratum
    stratification = Stratification("vaccination", all_strata, COMPARTMENTS)

    # Initial conditions, everyone unvaccinated
    pop_split = {stratum: 0. for stratum in all_strata}
    pop_split[Vaccination.UNVACCINATED] = 1.
    stratification.set_population_split(pop_split)

    # Immunity adjustments equivalent to history approach
    apply_immunity_to_strat(stratification, params, stratified_adjusters, Vaccination.UNVACCINATED)

    # Assign all the VoC infectious seed to the unvaccinated, because otherwise we might be assigning VoC entries to a
    # stratum that supposed to be empty (i.e. because vaccination hasn't started yet)
    if params.voc_emergence:
        for voc_name, voc_values in params.voc_emergence.items():
            seed_split = {stratum: Multiply(0.) for stratum in all_strata}
            seed_split[Vaccination.UNVACCINATED] = Multiply(1.)
            stratification.set_flow_adjustments(f"seed_voc_{voc_name}", seed_split)

    return stratification
