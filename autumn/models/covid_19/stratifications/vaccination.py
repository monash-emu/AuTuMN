from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, INFECTION
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.vaccination import find_vaccine_action, get_hosp_given_case_effect
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)


def get_vaccination_strat(params: Parameters, vacc_strata: List, is_dosing_active: bool) -> Stratification:
    """
    This vaccination stratification ist three strata applied to all compartments of the model.
    First create the stratification object and split the starting population.
    """

    vacc_strat = Stratification("vaccination", vacc_strata, COMPARTMENTS)

    # Everyone starts out unvaccinated
    pop_split = {stratum: 0. for stratum in vacc_strata}
    pop_split[Vaccination.UNVACCINATED] = 1.
    vacc_strat.set_population_split(pop_split)

    """
    Preliminary processing.
    """

    infection_efficacy, severity_efficacy, symptomatic_adjuster, hospital_adjuster, ifr_adjuster, \
        vaccination_effects = {}, {}, {}, {}, {}, {}

    # Get vaccination effect parameters in the form needed for the model
    for stratum in vacc_strata[1:]:

        # Parameters to directly pull out
        effectiveness_keys = ["vacc_prop_prevent_infection", "overall_efficacy"]
        stratum_vacc_params = getattr(params.vaccination, stratum)
        if stratum_vacc_params.vacc_reduce_death:
            effectiveness_keys.append("vacc_reduce_death")
        vaccination_effects[stratum] = {key: getattr(stratum_vacc_params, key) for key in effectiveness_keys}

        # Parameters that need to be processed
        vaccination_effects[stratum]["infection_efficacy"], \
            vaccination_effects[stratum]["severity_efficacy"] = find_vaccine_action(
            vaccination_effects[stratum]["vacc_prop_prevent_infection"],
            vaccination_effects[stratum]["overall_efficacy"],
        )
        if stratum_vacc_params.vacc_reduce_hospitalisation:
            vaccination_effects[stratum]["vacc_reduce_hosp_given_case"] = get_hosp_given_case_effect(
                stratum_vacc_params.vacc_reduce_hospitalisation,
                vaccination_effects[stratum]["overall_efficacy"],
            )

        severity_adjustment = 1. - vaccination_effects[stratum]["severity_efficacy"]

        symptomatic_adjuster[stratum] = severity_adjustment

        # Use the standard severity adjustment if no specific request for reducing death
        ifr_adjuster[stratum] = 1. - vaccination_effects[stratum]["vacc_reduce_death"] if \
            "vacc_reduce_death" in vaccination_effects[stratum] else severity_adjustment
        hospital_adjuster[stratum] = 1. - vaccination_effects[stratum]["vacc_reduce_hosp_given_case"] if \
            "vacc_reduce_hospitalisation" in vaccination_effects[stratum] else 1.

        # Apply the calibration adjusters
        symptomatic_adjuster[stratum] *= params.clinical_stratification.props.symptomatic.multiplier
        ifr_adjuster[stratum] *= params.infection_fatality.multiplier

    """
    Vaccination effect against severe outcomes.
    """

    # Add the clinical adjustments parameters as overwrites in the same way as for history stratification
    adjs = get_all_adjustments(
        params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
        params.sojourn, ifr_adjuster[Vaccination.ONE_DOSE_ONLY], symptomatic_adjuster[Vaccination.ONE_DOSE_ONLY],
        hospital_adjuster[Vaccination.ONE_DOSE_ONLY]
    )
    flow_adjs = get_blank_adjustments_for_strat(Vaccination.UNVACCINATED)
    flow_adjs = update_adjustments_for_strat(Vaccination.ONE_DOSE_ONLY, flow_adjs, adjs)

    # Apply for the fully vaccinated if active
    if is_dosing_active:
        second_adjs = \
            get_all_adjustments(
                params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
                params.sojourn, ifr_adjuster[Vaccination.VACCINATED], symptomatic_adjuster[Vaccination.VACCINATED],
                hospital_adjuster[Vaccination.VACCINATED], params.infection_fatality.top_bracket_overwrite,
            )
        flow_adjs = update_adjustments_for_strat(Vaccination.VACCINATED, flow_adjs, second_adjs)

    # Apply to the stratification object
    vacc_strat = add_clinical_adjustments_to_strat(vacc_strat, flow_adjs)

    """
    Vaccination effect against infection.
    """

    infection_adjustments = {
        stratum: Multiply(1. - vaccination_effects[stratum]["infection_efficacy"]) for stratum in vacc_strata[1:]
    }
    infection_adjustments.update({Vaccination.UNVACCINATED: None})
    vacc_strat.add_flow_adjustments(INFECTION, infection_adjustments)

    """
    Vaccination effect against infectiousness.
    """

    infectiousness_adjustments = {
        Vaccination.UNVACCINATED: None,
        Vaccination.ONE_DOSE_ONLY: Multiply(1. - params.vaccination.one_dose.vacc_reduce_infectiousness),
    }
    if is_dosing_active:
        infectiousness_adjustments.update({
            Vaccination.VACCINATED: Multiply(1. - params.vaccination.fully_vaccinated.vacc_reduce_infectiousness)
        })

    for compartment in DISEASE_COMPARTMENTS:
        vacc_strat.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    return vacc_strat
