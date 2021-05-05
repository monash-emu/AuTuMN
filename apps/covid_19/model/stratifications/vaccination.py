from summer import Multiply, Stratification

from apps.covid_19.constants import COMPARTMENTS, Clinical
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.vaccination import add_clinical_adjustments_to_strat

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
    relative_severity_effect = 1.

    # Everyone starts out unvaccinated.
    immunity_strat.set_population_split({"unvaccinated": 1., "vaccinated": 0.})

    # Sort out the parameters to be applied.
    relative_severity_effect -= params.vaccination.severity_efficacy
    symptomatic_adjuster = (params.clinical_stratification.props.symptomatic.multiplier * relative_severity_effect)
    hospital_adjuster = params.clinical_stratification.props.hospital.multiplier
    ifr_adjuster = params.infection_fatality.multiplier * relative_severity_effect

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
            "vaccinated": Multiply(1. - params.vaccination.infection_efficacy),
            "unvaccinated": None,
        },
    )

    return immunity_strat
