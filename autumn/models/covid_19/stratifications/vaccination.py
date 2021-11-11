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

    # Preliminary processing
    infection_effect, severity_effect, symptomatic_adjuster, hospital_adjuster, ifr_adjuster = {}, {}, {}, {}, {}
    vaccination_effects = get_vacc_effects_by_stratum(symptomatic_adjuster, hospital_adjuster, ifr_adjuster, params)

    # Vaccination effect against severe outcomes
    flow_adjs = get_blank_adjustments_for_strat(unadjusted_strata)
    for stratum in vacc_strata:
        adjs = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, ifr_adjuster[stratum], symptomatic_adjuster[stratum], hospital_adjuster[stratum]
        )
        flow_adjs = update_adjustments_for_strat(stratum, flow_adjs, adjs)

    stratification = add_clinical_adjustments_to_strat(stratification, flow_adjs)

    # Vaccination effect against infection
    infection_adjustments = {stratum: None for stratum in unadjusted_strata}
    adjs = {strat: Multiply(1. - vaccination_effects[strat]["infection_efficacy"]) for strat in vacc_strata}
    infection_adjustments.update(adjs)
    stratification.add_flow_adjustments(INFECTION, infection_adjustments)

    # Vaccination effect against infectiousness
    infectiousness_adjustments = {stratum: None for stratum in unadjusted_strata}
    adjs = {strat: Multiply(1. - getattr(getattr(vacc_params, strat), "ve_infectiousness")) for strat in vacc_strata}
    infectiousness_adjustments.update(adjs)
    for compartment in DISEASE_COMPARTMENTS:
        stratification.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    return stratification
