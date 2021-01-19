from typing import List

import numpy as np

from summer2 import Stratification, Multiply, Overwrite
from autumn.curve import scale_up_function
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.case_detection import build_detected_proportion_func
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from apps.covid_19.constants import (
    Compartment,
    Clinical,
    INFECTIOUS_COMPARTMENTS,
)
from apps.covid_19.model.preprocess.clinical import (
    get_proportion_symptomatic,
    get_abs_prop_isolated_factory,
    get_abs_prop_sympt_non_hospital_factory,
    get_absolute_strata_proportions,
    get_absolute_death_proportions,
    get_infection_fatality_proportions,
)

CLINICAL_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.SYMPT_NON_HOSPITAL,
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]


def get_clinical_strat(params: Parameters) -> Stratification:

    """
    Stratify the model by clinical status
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)

    - notifications are derived from progress from early to late for some strata
    - the proportion of people moving from presympt to early infectious, conditioned on age group
    - rate of which people flow through these compartments (reciprocal of time, using within_* which is a rate of ppl / day)
    - infectiousness levels adjusted by early/late and for clinical strata
    - we start with an age stratified infection fatality rate
        - 50% of deaths for each age bracket die in ICU
        - the other deaths go to hospital, assume no-one else can die from COVID
        - should we ditch this?

    """
    clinical_strat = Stratification("clinical", CLINICAL_STRATA, INFECTIOUS_COMPARTMENTS)
    clinical_params = params.clinical_stratification
    country = params.country
    pop = params.population

    """
    Infectiousness adjustments for clinical strat
    """
    # Add infectiousness reduction multiplier for all non-symptomatic infectious people.
    # These people are less infectious because of biology.
    non_sympt_adjust = Overwrite(clinical_params.non_sympt_infect_multiplier)
    clinical_strat.add_infectiousness_adjustments(
        Compartment.LATE_EXPOSED,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: None,
            Clinical.ICU: None,
        },
    )
    clinical_strat.add_infectiousness_adjustments(
        Compartment.EARLY_ACTIVE,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: None,
            Clinical.ICU: None,
        },
    )
    # Add infectiousness reduction for people who are late active and in isolation or hospital/icu.
    # These peoplee are less infectious because of physical distancing/isolation/PPE precautions.
    late_infect_multiplier = clinical_params.late_infect_multiplier
    clinical_strat.add_infectiousness_adjustments(
        Compartment.LATE_ACTIVE,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_ISOLATE: Overwrite(late_infect_multiplier[Clinical.SYMPT_ISOLATE]),
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.HOSPITAL_NON_ICU: Overwrite(late_infect_multiplier[Clinical.HOSPITAL_NON_ICU]),
            Clinical.ICU: Overwrite(late_infect_multiplier[Clinical.ICU]),
        },
    )

    """
    Adjust infection death rates for hospital patients (ICU and non-ICU)
    """

    # Proportion of people in age group who die, given the number infected: dead / total infected.
    infection_fatality = params.infection_fatality
    infection_fatality_props = get_infection_fatality_proportions(
        infection_fatality_props_10_year=infection_fatality.props,
        infection_rate_multiplier=infection_fatality.multiplier,
        iso3=country.iso3,
        pop_region=pop.region,
        pop_year=pop.year,
    )

    # Get the proportion of people in each clinical stratum, relative to total people in compartment.
    symptomatic_props = get_proportion_symptomatic(params)
    abs_props = get_absolute_strata_proportions(
        symptomatic_props=symptomatic_props,
        icu_props=clinical_params.icu_prop,
        hospital_props=clinical_params.props.hospital.props,
        symptomatic_props_multiplier=clinical_params.props.symptomatic.multiplier,
        hospital_props_multiplier=clinical_params.props.hospital.multiplier,
    )

    # Get the proportion of people who die for each strata/agegroup, relative to total infected.
    abs_death_props = get_absolute_death_proportions(
        abs_props=abs_props,
        infection_fatality_props=infection_fatality_props,
        icu_mortality_prop=clinical_params.icu_mortality_prop,
    )

    # Calculate relative death proportions for each strata / agegroup.
    # This is the number of people in strata / agegroup who die, given the total num people in that strata / agegroup.
    relative_death_props = {
        stratum: np.array(abs_death_props[stratum]) / np.array(abs_props[stratum])
        for stratum in (
            Clinical.HOSPITAL_NON_ICU,
            Clinical.ICU,
            Clinical.NON_SYMPT,
        )
    }

    # Now we want to convert these death proprotions into flow rates.
    # These flow rates are the death rates for hospitalised patients in ICU and non-ICU.
    # We assume everyone who dies does so at the end of their time in the "late active" compartment.
    # We split the flow rate out of "late active" into a death or recovery flow, based on the relative death proportion.
    sojourn = params.sojourn
    within_hospital_late = 1.0 / sojourn.compartment_periods["hospital_late"]
    within_icu_late = 1.0 / sojourn.compartment_periods["icu_late"]
    hospital_death_rates = relative_death_props[Clinical.HOSPITAL_NON_ICU] * within_hospital_late
    icu_death_rates = relative_death_props[Clinical.ICU] * within_icu_late

    # Apply adjusted infection death rates for hospital patients (ICU and non-ICU)
    # Death and non-death progression between infectious compartments towards the recovered compartment
    for idx, agegroup in enumerate(AGEGROUP_STRATA):
        death_adjs = {
            Clinical.NON_SYMPT: None,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_death_rates[idx]),
            Clinical.ICU: Overwrite(icu_death_rates[idx]),
        }
        clinical_strat.add_flow_adjustments(
            "infect_death", death_adjs, source_strata={"agegroup": agegroup}
        )

    """
    Adjust early exposed sojourn times.
    """
    # Progression rates into the infectious compartment(s)
    # Define progression rates into non-symptomatic compartments using parameter adjustment.
    # Get case detection rate function.
    get_detected_proportion = build_detected_proportion_func(
        AGEGROUP_STRATA, country, pop, params.testing_to_detection, params.case_detection
    )

    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        get_abs_prop_isolated = get_abs_prop_isolated_factory(
            age_idx, abs_props, get_detected_proportion
        )
        get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )
        adjustments = {
            Clinical.NON_SYMPT: Multiply(abs_props[Clinical.NON_SYMPT][age_idx]),
            Clinical.ICU: Multiply(abs_props[Clinical.ICU][age_idx]),
            Clinical.HOSPITAL_NON_ICU: Multiply(abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]),
            Clinical.SYMPT_NON_HOSPITAL: Multiply(get_abs_prop_sympt_non_hospital),
            Clinical.SYMPT_ISOLATE: Multiply(get_abs_prop_isolated),
        }
        clinical_strat.add_flow_adjustments(
            "infect_onset",
            adjustments,
            source_strata={"agegroup": agegroup},
        )

    """
    Adjust early active sojourn times.
    """
    within_hospital_early = 1.0 / sojourn.compartment_periods["hospital_early"]
    within_icu_early = 1.0 / sojourn.compartment_periods["icu_early"]
    for agegroup in AGEGROUP_STRATA:
        clinical_strat.add_flow_adjustments(
            "progress",
            {
                Clinical.NON_SYMPT: None,
                Clinical.ICU: Overwrite(within_icu_early),
                Clinical.HOSPITAL_NON_ICU: Overwrite(within_hospital_early),
                Clinical.SYMPT_NON_HOSPITAL: None,
                Clinical.SYMPT_ISOLATE: None,
            },
            source_strata={"agegroup": agegroup},
        )

    """
    Adjust late active sojourn times.
    """
    hospital_survival_props = 1 - relative_death_props[Clinical.HOSPITAL_NON_ICU]
    icu_survival_props = 1 - relative_death_props[Clinical.ICU]
    hospital_survival_rates = within_hospital_late * hospital_survival_props
    icu_survival_rates = within_icu_late * icu_survival_props

    for idx, agegroup in enumerate(AGEGROUP_STRATA):
        clinical_strat.add_flow_adjustments(
            "recovery",
            {
                Clinical.NON_SYMPT: None,
                Clinical.ICU: Overwrite(icu_survival_rates[idx]),
                Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_survival_rates[idx]),
                Clinical.SYMPT_NON_HOSPITAL: None,
                Clinical.SYMPT_ISOLATE: None,
            },
            source_strata={"agegroup": agegroup},
        )

    """
    Clinical proportions for imported cases.
    """
    # Work out time-variant clinical proportions for imported cases accounting for quarantine
    importation = params.importation
    if importation:
        # Create scale-up function for quarantine. Default to no quarantine if not values are available.
        qts = importation.quarantine_timeseries
        quarantine_func = scale_up_function(qts.times, qts.values, method=4) if qts else lambda _: 0

        # Loop through age groups and set the appropriate clinical proportions
        for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
            # Proportion entering non-symptomatic stratum reduced by the quarantined (and so isolated) proportion
            def get_prop_imported_to_nonsympt(t):
                prop_not_quarantined = 1.0 - quarantine_func(t)
                abs_prop_nonsympt = abs_props[Clinical.NON_SYMPT][age_idx]
                return abs_prop_nonsympt * prop_not_quarantined

            get_abs_prop_isolated = get_abs_prop_isolated_factory(
                age_idx, abs_props, get_detected_proportion
            )
            get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
                age_idx, abs_props, get_abs_prop_isolated
            )

            # Proportion ambulatory also reduced by quarantined proportion due to isolation
            def get_prop_imported_to_sympt_non_hospital(t):
                prop_not_quarantined = 1.0 - quarantine_func(t)
                abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital(t)
                return abs_prop_sympt_non_hospital * prop_not_quarantined

            # Proportion isolated includes those that would have been detected anyway and the ones above quarantined
            def get_prop_imported_to_sympt_isolate(t):
                abs_prop_isolated = get_abs_prop_isolated(t)
                prop_quarantined = quarantine_func(t)
                abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital(t)
                abs_prop_nonsympt = abs_props[Clinical.NON_SYMPT][age_idx]
                return abs_prop_isolated + prop_quarantined * (
                    abs_prop_sympt_non_hospital + abs_prop_nonsympt
                )

            clinical_strat.add_flow_adjustments(
                "importation",
                {
                    Clinical.NON_SYMPT: Multiply(get_prop_imported_to_nonsympt),
                    Clinical.ICU: Multiply(abs_props[Clinical.ICU][age_idx]),
                    Clinical.HOSPITAL_NON_ICU: Multiply(
                        abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
                    ),
                    Clinical.SYMPT_NON_HOSPITAL: Multiply(get_prop_imported_to_sympt_non_hospital),
                    Clinical.SYMPT_ISOLATE: Multiply(get_prop_imported_to_sympt_isolate),
                },
                dest_strata={"agegroup": agegroup},
            )

    return clinical_strat
