from summer import Multiply, Stratification

from autumn.models.covid_19.constants import (
    COMPARTMENTS, DISEASE_COMPARTMENTS, Vaccination, VACCINATION_STRATA, INFECTION
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.vaccination import \
    add_clinical_adjustments_to_strat, add_vaccine_infection_and_severity


def get_vaccination_strat(params: Parameters) -> Stratification:
    """
    Create the vaccination stratification as two strata applied to the whole model.
    """

    vacc_strat = Stratification("vaccination", VACCINATION_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated
    vacc_strat.set_population_split({Vaccination.UNVACCINATED: 1., Vaccination.VACCINATED: 0.})

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
        vacc_strat,
        Vaccination.UNVACCINATED,
        Vaccination.VACCINATED,
        params,
        symptomatic_adjuster,
        hospital_adjuster,
        ifr_adjuster,
        params.infection_fatality.top_bracket_overwrite,
    )

    # Apply vaccination effect against transmission
    vacc_strat.add_flow_adjustments(
        INFECTION,
        {
            Vaccination.UNVACCINATED: None,
            Vaccination.VACCINATED: Multiply(1. - infection_efficacy),
        },
    )

    # Adjust the infectiousness of vaccinated people
    for compartment in DISEASE_COMPARTMENTS:
        vacc_strat.add_infectiousness_adjustments(
            compartment,
            {
                Vaccination.UNVACCINATED: None,
                Vaccination.VACCINATED: Multiply(1. - params.vaccination.vacc_reduce_infectiousness),
            }
        )

    return vacc_strat
