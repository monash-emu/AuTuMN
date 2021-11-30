from typing import Dict

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION
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

    # Preliminaries
    vacc_params = params.vaccination
    modified_strata = all_strata[1:]  # The affected strata are all but the first, which is the unvaccinated

    vacc_effects = apply_immunity_to_strat(stratification, params, stratified_adjusters, Vaccination.UNVACCINATED)

    # Vaccination effect against infection
    infect_adjs = {stratum: Multiply(1. - vacc_effects[stratum]["infection_efficacy"]) for stratum in modified_strata}
    infect_adjs.update({Vaccination.UNVACCINATED: None})
    stratification.add_flow_adjustments(INFECTION, infect_adjs)

    # Vaccination effect against infectiousness
    infectiousness_adjs = {s: Multiply(1. - getattr(getattr(vacc_params, s), "ve_infectiousness")) for s in modified_strata}
    infectiousness_adjs.update({Vaccination.UNVACCINATED: None})
    for compartment in DISEASE_COMPARTMENTS:
        stratification.add_infectiousness_adjustments(compartment, infectiousness_adjs)

    # Simplest approach for VoCs is to assign all the VoC infectious seed to the unvaccinated
    # FIXME: This can probably be deleted, once the summer importations split is fixed
    if params.voc_emergence:
        for voc_name, voc_values in params.voc_emergence.items():
            seed_split = {stratum: Multiply(0.) for stratum in modified_strata}
            seed_split[Vaccination.UNVACCINATED] = Multiply(1.)
            stratification.add_flow_adjustments(f"seed_voc_{voc_name}", seed_split)

    return stratification
