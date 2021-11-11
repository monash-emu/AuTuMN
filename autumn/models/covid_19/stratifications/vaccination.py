from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)
from autumn.models.covid_19.strat_processing.vaccination import get_vacc_effects_by_stratum


def get_vaccination_strat(params: Parameters, all_strata: List) -> Stratification:
    """
    This vaccination stratification ist three strata applied to all compartments of the model.
    First create the stratification object and split the starting population.
    """

    vacc_params = params.vaccination

    # Create the stratum
    stratification = Stratification("vaccination", all_strata, COMPARTMENTS)

    # Unaffected and affected strata
    unadjusted_strata = all_strata[: 1]  # Slice to create single element list
    vacc_strata = all_strata[1:]

    # Initial conditions, everyone unvaccinated
    pop_split = {stratum: 0. for stratum in all_strata}
    pop_split[Vaccination.UNVACCINATED] = 1.
    stratification.set_population_split(pop_split)

    # Preliminaries
    vaccination_effects, symptomatic_adjusters, hospital_adjusters, ifr_adjusters = get_vacc_effects_by_stratum(params)

    # Vaccination effect against severe outcomes
    flow_adjs = get_blank_adjustments_for_strat(unadjusted_strata)
    for stratum in vacc_strata:
        severity_adjs = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, ifr_adjusters[stratum], symptomatic_adjusters[stratum], hospital_adjusters[stratum]
        )
        flow_adjs = update_adjustments_for_strat(stratum, flow_adjs, severity_adjs)
    stratification = add_clinical_adjustments_to_strat(stratification, flow_adjs)

    # Vaccination effect against infection
    infection_adjustments = {stratum: None for stratum in unadjusted_strata}
    infect_adjs = {strat: Multiply(1. - vaccination_effects[strat]["infection_efficacy"]) for strat in vacc_strata}
    infection_adjustments.update(infect_adjs)
    stratification.add_flow_adjustments(INFECTION, infection_adjustments)

    # Vaccination effect against infectiousness
    infectiousness_adjustments = {stratum: None for stratum in unadjusted_strata}
    infectiousness_adjs = {s: Multiply(1. - getattr(getattr(vacc_params, s), "ve_infectiousness")) for s in vacc_strata}
    infectiousness_adjustments.update(infectiousness_adjs)
    for compartment in DISEASE_COMPARTMENTS:
        stratification.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    # Simplest approach for VoCs is to assign all the VoC infectious seed to the unvaccinated
    if params.voc_emergence:
        for voc_name, voc_values in params.voc_emergence.items():
            seed_split = {stratum: Multiply(0.) for stratum in vacc_strata}
            seed_split[Vaccination.UNVACCINATED] = Multiply(1.)
            stratification.add_flow_adjustments(f"seed_voc_{voc_name}", seed_split)

    return stratification
