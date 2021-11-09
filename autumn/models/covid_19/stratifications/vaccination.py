from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION, VACCINATION_STRATA
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.vaccination import find_vaccine_action, get_hosp_given_case_effect
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)


def get_vaccination_strat(
        params: Parameters, vacc_strata: List, is_waning_vacc_immunity: bool
) -> Stratification:
    """
    This vaccination stratification ist three strata applied to all compartments of the model.
    First create the stratification object and split the starting population.
    """

    # Need to change this - currently a place-holder
    vaccinated_strata = vacc_strata[1: 3]

    vacc_strat = Stratification("vaccination", vacc_strata, COMPARTMENTS)

    # Everyone starts out unvaccinated
    pop_split = {stratum: 0. for stratum in vacc_strata}
    pop_split[Vaccination.UNVACCINATED] = 1.
    vacc_strat.set_population_split(pop_split)

    """
    Preliminary processing.
    """

    infection_effect, severity_effect, symptomatic_adjuster, hospital_adjuster, ifr_adjuster, \
        vaccination_effects = {}, {}, {}, {}, {}, {}

    # Get vaccination effect parameters in the form needed for the model
    for stratum in vaccinated_strata:

        # Parameters to directly pull out
        raw_effectiveness_keys = ["ve_prop_prevent_infection", "ve_sympt_covid"]
        stratum_vacc_params = getattr(params.vaccination, stratum)
        if stratum_vacc_params.ve_death:
            raw_effectiveness_keys.append("ve_death")
        vaccination_effects[stratum] = {key: getattr(stratum_vacc_params, key) for key in raw_effectiveness_keys}

        # Parameters that need to be processed
        vaccination_effects[stratum]["infection_efficacy"], severity_effect = find_vaccine_action(
            vaccination_effects[stratum]["ve_prop_prevent_infection"],
            vaccination_effects[stratum]["ve_sympt_covid"],
        )
        if stratum_vacc_params.ve_hospitalisation:
            hospitalisation_effect = get_hosp_given_case_effect(
                stratum_vacc_params.ve_hospitalisation, vaccination_effects[stratum]["ve_sympt_covid"],
            )

        symptomatic_adjuster[stratum] = 1. - severity_effect

        # Use the standard severity adjustment if no specific request for reducing death
        ifr_adjuster[stratum] = 1. - vaccination_effects[stratum]["ve_death"] if \
            "ve_death" in vaccination_effects[stratum] else 1. - severity_effect
        hospital_adjuster[stratum] = 1. - hospitalisation_effect if \
            "ve_hospitalisation" in vaccination_effects[stratum] else 1.

        # Apply the calibration adjusters
        symptomatic_adjuster[stratum] *= params.clinical_stratification.props.symptomatic.multiplier
        ifr_adjuster[stratum] *= params.infection_fatality.multiplier

    """
    Vaccination effect against severe outcomes.
    """

    unadjusted_strata = [Vaccination.UNVACCINATED]
    if is_waning_vacc_immunity:
        unadjusted_strata += VACCINATION_STRATA[3:]
    flow_adjs = get_blank_adjustments_for_strat(unadjusted_strata)

    # Add the clinical adjustments parameters as overwrites in the same way as for history stratification
    # flow_adjs = get_blank_adjustments_for_strat([Vaccination.UNVACCINATED])

    for stratum in vaccinated_strata:
        adjs = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, ifr_adjuster[stratum], symptomatic_adjuster[stratum], hospital_adjuster[stratum]
        )
        flow_adjs = update_adjustments_for_strat(stratum, flow_adjs, adjs)

    # Apply to the stratification object
    vacc_strat = add_clinical_adjustments_to_strat(vacc_strat, flow_adjs)

    """
    Vaccination effect against infection.
    """

    infection_adjustments = {stratum: None for stratum in unadjusted_strata}
    strata_adjs = {
        stratum: Multiply(1. - vaccination_effects[stratum]["infection_efficacy"]) for stratum in vaccinated_strata
    }
    infection_adjustments.update(strata_adjs)
    vacc_strat.add_flow_adjustments(INFECTION, infection_adjustments)

    """
    Vaccination effect against infectiousness.
    """

    infectiousness_adjustment_strata = VACCINATION_STRATA[1:]

    infectiousness_adjustments = {stratum: None for stratum in unadjusted_strata}
    strata_adjs = {
        stratum: Multiply(1. - getattr(getattr(params.vaccination, stratum), "ve_infectiousness")) for
        stratum in infectiousness_adjustment_strata
    }
    infectiousness_adjustments.update(strata_adjs)
    for compartment in DISEASE_COMPARTMENTS:
        vacc_strat.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    return vacc_strat
