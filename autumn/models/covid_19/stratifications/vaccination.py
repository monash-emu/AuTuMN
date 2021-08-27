from summer import Multiply, Stratification

from autumn.models.covid_19.constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, VACCINATION_STRATA, INFECTION, VACCINATED_CATEGORIES
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.vaccination import (
    add_clinical_adjustments_to_strat, add_vaccine_infection_and_severity
)


def get_vaccination_strat(params: Parameters) -> Stratification:
    """
    Create the vaccination stratification as two strata applied to the whole model.
    """

    # Apply requested vaccination strata to all of the model's compartments
    vacc_strat = Stratification("vaccination", VACCINATION_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated
    starting_splits = {Vaccination.UNVACCINATED: 1.}
    starting_splits.update({stratum: 0. for stratum in VACCINATED_CATEGORIES})
    vacc_strat.set_population_split(starting_splits)

    # Process the parameters to be applied
    infection_efficacy, severity_efficacy = add_vaccine_infection_and_severity(
        params.vaccination.vacc_prop_prevent_infection,
        params.vaccination.overall_efficacy
    )
    symptomatic_adjuster, hospital_adjuster, ifr_adjuster = (1. - severity_efficacy,) * 3

    # Apply the calibration adjustment parameters
    symptomatic_adjuster *= params.clinical_stratification.props.symptomatic.multiplier
    ifr_adjuster *= params.infection_fatality.multiplier

    # Add the clinical adjustments parameters as overwrites in the same way as for vaccination
    vacc_strat = add_clinical_adjustments_to_strat(
        vacc_strat, Vaccination.UNVACCINATED, VACCINATED_CATEGORIES, params, symptomatic_adjuster, hospital_adjuster,
        ifr_adjuster, params.infection_fatality.top_bracket_overwrite,
    )

    # Apply vaccination effect against transmission
    transmission_adjustments = {Vaccination.UNVACCINATED: None}
    transmission_adjustments.update({stratum: Multiply(1. - infection_efficacy) for stratum in VACCINATED_CATEGORIES})
    vacc_strat.add_flow_adjustments(INFECTION, transmission_adjustments)

    # Adjust the infectiousness of vaccinated people
    for compartment in DISEASE_COMPARTMENTS:
        infectiousness_adjustments = {Vaccination.UNVACCINATED: None}
        rel_infectiousness = 1. - params.vaccination.vacc_reduce_infectiousness
        infectiousness_adjustments.update({stratum: Multiply(rel_infectiousness) for stratum in VACCINATED_CATEGORIES})
        vacc_strat.add_infectiousness_adjustments(compartment, infectiousness_adjustments)

    return vacc_strat
