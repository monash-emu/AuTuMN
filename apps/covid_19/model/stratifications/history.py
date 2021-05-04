from summer import Multiply, Stratification

from apps.covid_19.model.parameters import Parameters
from apps.covid_19.constants import COMPARTMENTS, Clinical

from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from apps.covid_19.model.stratifications.clinical import CLINICAL_STRATA
from apps.covid_19.model.preprocess.clinical import get_all_adjs


CLINICAL_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.SYMPT_NON_HOSPITAL,
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

HISTORY_STRATA = [
    "naive",
    "experienced",
]


def get_history_strat(params: Parameters) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.
    """
    history_strat = Stratification("history", HISTORY_STRATA, COMPARTMENTS)

    # Everyone starts out infection-naive.
    history_strat.set_population_split({"naive": 1.0, "experienced": 0.0})

    # Waning immunity makes recovered individuals transition to the 'experienced' stratum.
    if params.waning_immunity_duration is not None:
        history_strat.add_flow_adjustments(
            "waning_immunity", {"naive": Multiply(0.), "experienced": Multiply(1.)}
        )

    # Placeholder parameters for the effect of past infection on protection against severe disease given infection.
    symptomatic_adjuster, hospital_adjuster, ifr_adjuster = 1., 1., 1.

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
            history_strat.add_flow_adjustments(
                "infect_onset",
                {"naive": None, "experienced": entry_adjustments[agegroup][clinical_stratum]},
                dest_strata=relevant_strata,  # Must be dest
            )
            history_strat.add_flow_adjustments(
                "infect_death",
                {"naive": None, "experienced": death_adjs[agegroup][clinical_stratum]},
                source_strata=relevant_strata,  # Must be source
            )
            history_strat.add_flow_adjustments(
                "progress",
                {"naive": None, "experienced": progress_adjs[clinical_stratum]},
                source_strata=relevant_strata,  # Either source or dest or both
            )
            history_strat.add_flow_adjustments(
                "recovery",
                {"naive": None, "experienced": recovery_adjs[agegroup][clinical_stratum]},
                source_strata=relevant_strata,  # Must be source
            )

    return history_strat
