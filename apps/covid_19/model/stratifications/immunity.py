from summer2 import Stratification, Multiply, Overwrite
from apps.covid_19.constants import COMPARTMENTS, Clinical, Compartment
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model import preprocess
from apps.covid_19.model.preprocess.clinical import \
    get_absolute_death_proportions, \
    get_hosp_sojourns, \
    get_hosp_death_rates, \
    apply_death_adjustments, \
    get_entry_adjustments
from apps.covid_19.model.stratifications.clinical import get_ifr_props, get_sympt_props, get_relative_death_props
from apps.covid_19.model.preprocess.case_detection import build_detected_proportion_func


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

    vaccine_efficacy = 0.
    symptomatic_adjuster = \
        (1. - vaccine_efficacy) * \
        params.clinical_stratification.props.symptomatic.multiplier
    hospital_adjuster = \
        (1. - vaccine_efficacy) * \
        params.clinical_stratification.props.hospital.multiplier
    ifr_adjuster = \
        (1. - vaccine_efficacy) * \
        params.infection_fatality.multiplier
    infection_fatality_props = \
        get_ifr_props(
            params,
            ifr_adjuster
        )
    abs_props = \
        get_sympt_props(
            params,
            symptomatic_adjuster,
            hospital_adjuster,
        )
    abs_death_props = \
        get_absolute_death_proportions(abs_props, infection_fatality_props, clinical_params.icu_mortality_prop)
    relative_death_props = \
        get_relative_death_props(abs_props, abs_death_props)
    sojourn = \
        params.sojourn
    within_hospital_late, within_icu_late = \
        get_hosp_sojourns(sojourn)
    hospital_death_rates, icu_death_rates = \
        get_hosp_death_rates(relative_death_props, within_hospital_late, within_icu_late)
    death_adjs = \
        apply_death_adjustments(hospital_death_rates, icu_death_rates)
    get_detected_proportion = build_detected_proportion_func(
        AGEGROUP_STRATA, country, pop, params.testing_to_detection, params.case_detection
    )
    compartment_periods = \
        preprocess.compartments.calc_compartment_periods(params.sojourn)
    entry_adjustments = \
        get_entry_adjustments(
            abs_props,
            get_detected_proportion,
            1. / compartment_periods[Compartment.EARLY_EXPOSED])
    within_hospital_early = \
        1. / sojourn.compartment_periods["hospital_early"]
    within_icu_early = \
        1. / sojourn.compartment_periods["icu_early"]
    hospital_survival_props = \
        1. - relative_death_props[Clinical.HOSPITAL_NON_ICU]
    icu_survival_props = \
        1. - relative_death_props[Clinical.ICU]
    hospital_survival_rates = \
        within_hospital_late * hospital_survival_props
    icu_survival_rates = \
        within_icu_late * icu_survival_props

    progress_adjs = {
        Clinical.NON_SYMPT: None,
        Clinical.ICU: Overwrite(within_icu_early),
        Clinical.HOSPITAL_NON_ICU: Overwrite(within_hospital_early),
        Clinical.SYMPT_NON_HOSPITAL: None,
        Clinical.SYMPT_ISOLATE: None
    }

    for i_age, agegroup in enumerate(AGEGROUP_STRATA):
        for clinical_stratum in CLINICAL_STRATA:
            source = {
                "agegroup": agegroup,
                "clinical": clinical_stratum,
            }
            immunity_strat.add_flow_adjustments(
                "infect_onset",
                {
                    "unvaccinated": None,
                    "vaccinated": entry_adjustments[agegroup][clinical_stratum]
                },
                source_strata=source,
            )
            immunity_strat.add_flow_adjustments(
                "infect_death",
                {
                    "unvaccinated": None,
                    "vaccinated": death_adjs[agegroup][clinical_stratum]
                },
                source_strata=source
            )
            immunity_strat.add_flow_adjustments(
                "progress",
                {
                    "unvaccinated": None,
                    "vaccinated": progress_adjs[clinical_stratum]
                },
                source_strata=source
            )
            recovery_adjs = \
                {
                    Clinical.NON_SYMPT: None,
                    Clinical.ICU: Overwrite(icu_survival_rates[i_age]),
                    Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_survival_rates[i_age]),
                    Clinical.SYMPT_NON_HOSPITAL: None,
                    Clinical.SYMPT_ISOLATE: None
                }
            immunity_strat.add_flow_adjustments(
                "recovery",
                {
                    "unvaccinated": None,
                    "vaccinated": recovery_adjs[clinical_stratum]
                },
                source_strata=source
            )

    return immunity_strat
