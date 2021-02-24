from summer2 import Stratification, Multiply, Overwrite
from apps.covid_19.constants import COMPARTMENTS, Clinical
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.clinical import get_all_adjs


CLINICAL_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.SYMPT_NON_HOSPITAL,
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

IMMUNITY_STRATA = [
    "unvaccinated",
    "vaccinated",
]


def get_immunity_strat(params: Parameters) -> Stratification:
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated
    immunity_strat.set_population_split(
        {
            "unvaccinated": 1.0,
            "vaccinated": 0.0
        }
    )

    if params.vaccination:
        immunity_strat.add_flow_adjustments(
            "infection", {
                "vaccinated": Multiply(1. - params.vaccination.efficacy),
                "unvaccinated": None,
            }
        )

    clinical_params = params.clinical_stratification
    country = params.country
    pop = params.population

    vaccine_efficacy = 0.8  
    symptomatic_adjuster = \
        (1. - vaccine_efficacy) * \
        params.clinical_stratification.props.symptomatic.multiplier
    hospital_adjuster = \
        (1. - vaccine_efficacy) * \
        params.clinical_stratification.props.hospital.multiplier
    ifr_adjuster = \
        (1. - vaccine_efficacy) * \
        params.infection_fatality.multiplier

    # Get all the adjustments in the same way as we did for the clinical stratification
    entry_adjustments, death_adjs, progress_adjs, recovery_adjs, _, _ = \
        get_all_adjs(
            clinical_params,
            country,
            pop,
            params.infection_fatality.props,
            params.sojourn,
            params.testing_to_detection,
            params.case_detection,
            ifr_adjuster,
            symptomatic_adjuster,
            hospital_adjuster,
        )

    for i_age, agegroup in enumerate(AGEGROUP_STRATA):
        for clinical_stratum in CLINICAL_STRATA:
            relevant_strata = {
                "agegroup": agegroup,
                "clinical": clinical_stratum,
            }
            immunity_strat.add_flow_adjustments(
                "infect_onset",
                {
                    "unvaccinated": None,
                    "vaccinated": entry_adjustments[agegroup][clinical_stratum]
                },
                dest_strata=relevant_strata,  # Must be dest
            )
            immunity_strat.add_flow_adjustments(
                "infect_death",
                {
                    "unvaccinated": None,
                    "vaccinated": death_adjs[agegroup][clinical_stratum]
                },
                source_strata=relevant_strata,  # Must be source
            )
            immunity_strat.add_flow_adjustments(
                "progress",
                {
                    "unvaccinated": None,
                    "vaccinated": progress_adjs[clinical_stratum]
                },
                source_strata=relevant_strata,  # Either source or dest or both
            )
            immunity_strat.add_flow_adjustments(
                "recovery",
                {
                    "unvaccinated": None,
                    "vaccinated": recovery_adjs[agegroup][clinical_stratum]
                },
                source_strata=relevant_strata,  # Must be source
            )

    return immunity_strat
