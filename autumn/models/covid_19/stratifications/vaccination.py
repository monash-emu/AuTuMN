from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)
from autumn.models.covid_19.strat_processing.vaccination import get_stratum_vacc_effect


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
    ve_infection, ve_severity, sympt_adjusters, hosp_adjusters, ifr_adjusters, vacc_effects = {}, {}, {}, {}, {}, {}

    # Get vaccination effect parameters in the form needed for the model
    flow_adjs = get_blank_adjustments_for_strat(unadjusted_strata)
    for stratum in vacc_strata:
        vacc_effects[stratum], sympt_adjuster, hosp_adjuster, ifr_adjuster = get_stratum_vacc_effect(params, stratum)

        # Vaccination effect against severe outcomes
        sympt_adjuster *= params.clinical_stratification.props.symptomatic.multiplier
        ifr_adjuster *= params.infection_fatality.multiplier

        severity_adjs = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, ifr_adjuster, sympt_adjuster, hosp_adjuster
        )
        flow_adjs = update_adjustments_for_strat(stratum, flow_adjs, severity_adjs)
    stratification = add_clinical_adjustments_to_strat(stratification, flow_adjs)

    # Vaccination effect against infection
    infection_adjustments = {stratum: None for stratum in unadjusted_strata}
    infect_adjs = {stratum: Multiply(1. - vacc_effects[stratum]["infection_efficacy"]) for stratum in vacc_strata}
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
