from summer import Multiply, Stratification

from apps.covid_19.constants import COMPARTMENTS, Clinical
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.clinical import get_all_adjs
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA

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
    relative_severity_effect = 1.0

    # Everyone starts out unvaccinated
    immunity_strat.set_population_split({"unvaccinated": 1.0, "vaccinated": 0.0})

    if params.vaccination:

        # Apply vaccination effect against severe disease given infection
        relative_severity_effect -= params.vaccination.severity_efficacy

        # Apply vaccination effect against infection/transmission
        immunity_strat.add_flow_adjustments(
            "infection",
            {
                "vaccinated": Multiply(1.0 - params.vaccination.infection_efficacy),
                "unvaccinated": None,
            },
        )

        symptomatic_adjuster = (
            params.clinical_stratification.props.symptomatic.multiplier * relative_severity_effect
        )
        hospital_adjuster = params.clinical_stratification.props.hospital.multiplier
        ifr_adjuster = params.infection_fatality.multiplier * relative_severity_effect

        # Get all the adjustments in the same way as we did for the clinical stratification
        entry_adjustments, death_adjs, progress_adjs, recovery_adjs, _, _ = get_all_adjs(
            params.clinical_stratification,
            params.country,
            params.population,
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
                        "vaccinated": entry_adjustments[agegroup][clinical_stratum],
                    },
                    dest_strata=relevant_strata,  # Must be dest
                )
                immunity_strat.add_flow_adjustments(
                    "infect_death",
                    {"unvaccinated": None, "vaccinated": death_adjs[agegroup][clinical_stratum]},
                    source_strata=relevant_strata,  # Must be source
                )
                immunity_strat.add_flow_adjustments(
                    "progress",
                    {"unvaccinated": None, "vaccinated": progress_adjs[clinical_stratum]},
                    source_strata=relevant_strata,  # Either source or dest or both
                )
                immunity_strat.add_flow_adjustments(
                    "recovery",
                    {"unvaccinated": None, "vaccinated": recovery_adjs[agegroup][clinical_stratum]},
                    source_strata=relevant_strata,  # Must be source
                )

    return immunity_strat
