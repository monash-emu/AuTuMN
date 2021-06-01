from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, Clinical
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.vaccination import add_clinical_adjustments_to_strat
from autumn.models.covid_19.preprocess.vaccination import add_vaccine_infection_and_severity

CLINICAL_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.SYMPT_NON_HOSPITAL,
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

VACCINATION_STRATA = [
    "unvaccinated",
    "vaccinated",
]


def get_vaccination_strat(params: Parameters) -> Stratification:
    immunity_strat = Stratification("vaccination", VACCINATION_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated.
    immunity_strat.set_population_split({"unvaccinated": 1.0, "vaccinated": 0.0})

    # Sort out the parameters to be applied.

    infection_efficacy, severity_efficacy = add_vaccine_infection_and_severity(
        params.vaccination.vacc_prop_prevent_infection, params.vaccination.overall_efficacy
    )
    symptomatic_adjuster, hospital_adjuster, ifr_adjuster = (1.0 - severity_efficacy,) * 3

    # Apply the calibration adjustment parameters.
    symptomatic_adjuster *= params.clinical_stratification.props.symptomatic.multiplier
    ifr_adjuster *= params.infection_fatality.multiplier

    # Add the clinical adjustments parameters as overwrites in the same way as for vaccination.
    immunity_strat = add_clinical_adjustments_to_strat(
        immunity_strat,
        VACCINATION_STRATA[0],
        VACCINATION_STRATA[1],
        params,
        symptomatic_adjuster,
        hospital_adjuster,
        ifr_adjuster,
    )

    # Apply vaccination effect against infection/transmission
    immunity_strat.add_flow_adjustments(
        "infection",
        {
            "vaccinated": Multiply(1.0 - infection_efficacy),
            "unvaccinated": None,
        },
    )

    return immunity_strat
