from summer import Multiply, Stratification

from autumn.models.covid_19.constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, VACCINATION_STRATA, VACCINATED_STRATA, INFECTION
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.vaccination import find_vaccine_action
from autumn.models.covid_19.preprocess.clinical import add_clinical_adjustments_to_strat


def get_vaccination_strat(params: Parameters) -> Stratification:
    """
    This vaccination stratification ist three strata applied to all compartments of the model.
    First create the stratification object and split the starting population.
    """

    is_dosing_active = bool(params.vaccination.one_dose)
    vacc_strata = VACCINATION_STRATA if is_dosing_active else VACCINATION_STRATA[:2]

    vacc_strat = Stratification("vaccination", vacc_strata, COMPARTMENTS)

    # Everyone starts out unvaccinated
    pop_split = {Vaccination.UNVACCINATED: 1.}
    for stratum in vacc_strata[1:]:
        pop_split.update({stratum: 0.})
    vacc_strat.set_population_split(pop_split)

    """
    Preliminary processing.
    """

    strata_to_adjust = VACCINATED_STRATA if is_dosing_active else [Vaccination.ONE_DOSE_ONLY]
    infection_efficacy, severity_efficacy, symptomatic_adjuster, hospital_adjuster, ifr_adjuster = \
        {}, {}, {}, {}, {}

    # Get vaccination effect parameters in the form needed for the model
    one_dose_effects = {
        "prevent_infection": params.vaccination.one_dose.vacc_prop_prevent_infection,
        "overall_efficacy": params.vaccination.one_dose.overall_efficacy,
    }
    vaccination_effects = {Vaccination.ONE_DOSE_ONLY: one_dose_effects}

    # Get effects of two doses if implemented
    if is_dosing_active:
        full_vacc_effects = {
            "prevent_infection": params.vaccination.fully_vaccinated.vacc_prop_prevent_infection,
            "overall_efficacy": params.vaccination.fully_vaccinated.overall_efficacy,
        }
        vaccination_effects.update({Vaccination.VACCINATED: full_vacc_effects})

    # Get vaccination effects from requests by dose number and mode of action
    for stratum in strata_to_adjust:
        infection_efficacy[stratum], strat_severity_efficacy = find_vaccine_action(
            vaccination_effects[stratum]["prevent_infection"],
            vaccination_effects[stratum]["overall_efficacy"],
        )

        severity_adjustment = 1. - strat_severity_efficacy

        # Hospitalisation is risk given symptomatic case, so is de facto adjusted through symptomatic adjustment
        symptomatic_adjuster[stratum], hospital_adjuster[stratum], ifr_adjuster[stratum] = \
            severity_adjustment, 1., severity_adjustment

        # Apply the calibration adjustment parameters
        symptomatic_adjuster[stratum] *= params.clinical_stratification.props.symptomatic.multiplier
        ifr_adjuster[stratum] *= params.infection_fatality.multiplier

    """
    Vaccination effect against severe outcomes.
    """

    # Add the clinical adjustments parameters as overwrites in the same way as for history stratification
    vacc_strat = add_clinical_adjustments_to_strat(
        vacc_strat,
        Vaccination.UNVACCINATED,
        Vaccination.VACCINATED,
        params,
        symptomatic_adjuster[Vaccination.VACCINATED],
        hospital_adjuster[Vaccination.VACCINATED],
        ifr_adjuster[Vaccination.VACCINATED],
        params.infection_fatality.top_bracket_overwrite,
        second_modified_stratum=Vaccination.ONE_DOSE_ONLY,
        second_sympt_adjuster=symptomatic_adjuster[Vaccination.ONE_DOSE_ONLY] if is_dosing_active else 1.,
        second_hospital_adjuster=symptomatic_adjuster[Vaccination.ONE_DOSE_ONLY] if is_dosing_active else 1.,
        second_ifr_adjuster=symptomatic_adjuster[Vaccination.ONE_DOSE_ONLY] if is_dosing_active else 1.,
        second_top_bracket_overwrite=params.infection_fatality.top_bracket_overwrite,
    )

    """
    Vaccination effect against infection.
    """

    infection_adjustments = {
        Vaccination.UNVACCINATED: None,
        Vaccination.ONE_DOSE_ONLY: Multiply(1. - infection_efficacy[Vaccination.ONE_DOSE_ONLY])
    }
    if is_dosing_active:
        infection_adjustments.update(
            {Vaccination.VACCINATED: Multiply(1. - infection_efficacy[Vaccination.VACCINATED])}
        )

    vacc_strat.add_flow_adjustments(INFECTION, infection_adjustments)

    """
    Vaccination effect against infectiousness.
    """

    # These parameters can be used directly
    one_dose_infectious_adj = Multiply(1. - params.vaccination.one_dose.vacc_reduce_infectiousness)
    infectiousness_adjustments = {
        Vaccination.UNVACCINATED: None,
        Vaccination.ONE_DOSE_ONLY: one_dose_infectious_adj,
    }

    if is_dosing_active:
        fully_vacc_infectious_adj = Multiply(1. - params.vaccination.fully_vaccinated.vacc_reduce_infectiousness)
        infectiousness_adjustments.update(
            {Vaccination.VACCINATED: fully_vacc_infectious_adj}
        )

    for compartment in DISEASE_COMPARTMENTS:
        vacc_strat.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    return vacc_strat
