import numpy as np

from autumn.tool_kit.utils import (
    find_rates_and_complements_from_ifr,
    repeat_list_elements_average_last_two,
    element_wise_list_division,
)
from autumn.summer_related.parameter_adjustments import adjust_upstream_stratified_parameter
from autumn.curve import scale_up_function
from autumn.inputs import get_population_by_agegroup

from apps.covid_19.constants import Compartment, ClinicalStratum


def stratify_by_clinical(model, params, compartments, detected_proportion, symptomatic_props):
    """
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)

    - notifications are derived from progress from early to late for some strata
    - the proportion of people moving from presymt to early infectious, conditioned on age group
    - rate of which people flow through these compartments (reciprocal of time, using within_* which is a rate of ppl / day)
    - infectiousness levels adjusted by early/late and for clinical strata
    - we start with an age stratified infection fatality rate
        - 50% of deaths for each age bracket die in ICU
        - the other deaths go to hospital, assume no-one else can die from COVID
        - should we ditch this?

    """
    # General stratification
    agegroup_strata = model.stratifications[0].strata
    # Infection rate multiplication
    # Importation
    implement_importation = params["implement_importation"]
    traveller_quarantine = params["traveller_quarantine"]
    within_hospital_early = model.parameters["within_hospital_early"]
    within_icu_early = model.parameters["within_icu_early"]
    icu_prop = params["icu_prop"]
    icu_mortality_prop = params["icu_mortality_prop"]
    infection_fatality_props_10_year = params["infection_fatality_props"]
    hospital_props = params["hospital_props"]

    # Apply multiplier to symptomatic proportions
    symptomatic_props = [
        min([p * params["symptomatic_props_multiplier"], 1.0]) for p in symptomatic_props
    ]

    # Apply multiplier to hospital proportions
    hospital_props = [min([p * params["hospital_props_multiplier"], 1.0]) for p in hospital_props]

    # Define new strata.
    clinical_strata = [
        ClinicalStratum.NON_SYMPT,
        ClinicalStratum.SYMPT_NON_HOSPITAL,
        ClinicalStratum.SYMPT_ISOLATE,
        ClinicalStratum.HOSPITAL_NON_ICU,
        ClinicalStratum.ICU,
    ]
    non_hospital_strata = [
        ClinicalStratum.NON_SYMPT,
        ClinicalStratum.SYMPT_NON_HOSPITAL,
        ClinicalStratum.SYMPT_ISOLATE,
    ]
    hospital_strata = [
        ClinicalStratum.HOSPITAL_NON_ICU,
        ClinicalStratum.ICU,
    ]
    # Only stratify infected compartments
    compartments_to_stratify = [
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
    ]

    # FIXME: Set params to make comparison happy
    params["infection_fatality_props"] = [
        ifr * params["ifr_multiplier"] for ifr in infection_fatality_props_10_year
    ]

    # Calculate the proportion of 80+ years old among the 75+ population
    elderly_populations = get_population_by_agegroup([0, 75, 80], params["iso3"], params["region"])
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])

    # Infection fatality rate by age group.
    # Data in props used 10 year bands 0-80+, but we want 5 year bands from 0-75+

    # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
    infection_fatality_props = repeat_list_elements_average_last_two(
        infection_fatality_props_10_year, prop_over_80
    )

    # Find the absolute progression proportions.
    symptomatic_props_arr = np.array(symptomatic_props)
    hospital_props_arr = np.array(hospital_props)
    # Determine the absolute proportion of early exposed who become sympt vs non-sympt
    sympt, non_sympt = subdivide_props(1, symptomatic_props_arr)
    # Determine the absolute proportion of sympt who become hospitalized vs non-hospitalized.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, hospital_props_arr)
    # Determine the absolute proportion of hospitalized who become icu vs non-icu.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_prop)
    abs_props = {
        "sympt": sympt.tolist(),
        "non_sympt": non_sympt.tolist(),
        "hospital": sympt_hospital.tolist(),
        "sympt_non_hospital": sympt_non_hospital.tolist(),  # Over-ridden by a by time-varying proportion later
        ClinicalStratum.ICU: sympt_hospital_icu.tolist(),
        ClinicalStratum.HOSPITAL_NON_ICU: sympt_hospital_non_icu.tolist(),
    }

    # Calculate the absolute proportion of all patients who should eventually reach hospital death or ICU death.
    # Find IFR that needs to be contributed by ICU and non-ICU hospital deaths
    hospital_death, icu_death = [], []
    for age_idx, agegroup in enumerate(agegroup_strata):
        # If IFR for age group is greater than absolute proportion hospitalised, increased hospitalised proportion
        if infection_fatality_props[age_idx] > abs_props["hospital"][age_idx]:
            abs_props["hospital"][age_idx] = infection_fatality_props[age_idx]

        # Find the target absolute ICU mortality and the amount left over from IFRs to go to hospital, if any
        target_icu_abs_mort = abs_props[ClinicalStratum.ICU][age_idx] * icu_mortality_prop
        left_over_mort = infection_fatality_props[age_idx] - target_icu_abs_mort

        # If enough IFR left over to allow partial mortality for the hospitalised
        if 0.0 < left_over_mort <= abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]:
            hospital_death_prop = left_over_mort
            icu_death_prop = target_icu_abs_mort

        # Otherwise if there is too much excess death to fit in hospital, inflate the ICU proportion
        elif left_over_mort > abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]:
            hospital_death_prop = abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
            icu_death_prop = (
                infection_fatality_props[age_idx]
                - abs_props[ClinicalStratum.HOSPITAL_NON_ICU][age_idx]
            )

        # Otherwise if all IFR taken up by ICU
        else:
            hospital_death_prop = 0.0
            icu_death_prop = infection_fatality_props[age_idx]

        hospital_death.append(hospital_death_prop)
        icu_death.append(icu_death_prop)

    abs_props.update({"hospital_death": hospital_death, "icu_death": icu_death})

    # FIXME: These depend on static variables which have been made time-variant.
    # fatality rate for hospitalised patients
    rel_props = {
        "hospital_death": element_wise_list_division(
            abs_props["hospital_death"], abs_props[ClinicalStratum.HOSPITAL_NON_ICU]
        ),
        "icu_death": element_wise_list_division(
            abs_props["icu_death"], abs_props[ClinicalStratum.ICU]
        ),
    }

    # Progression rates into the infectious compartment(s)
    # Define progression rates into non-symptomatic compartments using parameter adjustment.
    flow_adjustments = {}
    for age_idx, age in enumerate(agegroup_strata):
        key = f"within_{Compartment.EARLY_EXPOSED}Xagegroup_{age}"
        flow_adjustments[key] = {
            ClinicalStratum.NON_SYMPT: non_sympt[age_idx],
            ClinicalStratum.ICU: sympt_hospital_icu[age_idx],
            ClinicalStratum.HOSPITAL_NON_ICU: sympt_hospital_non_icu[age_idx],
        }

    # Set time-varying isolation proportions
    for age_idx, agegroup in enumerate(agegroup_strata):
        # Pass the functions to the model
        tv_props = TimeVaryingProportions(age_idx, abs_props, detected_proportion)
        time_variants = [
            [
                f"prop_{ClinicalStratum.SYMPT_NON_HOSPITAL}_{agegroup}",
                tv_props.get_abs_prop_sympt_non_hospital,
            ],
            [f"prop_{ClinicalStratum.SYMPT_ISOLATE}_{agegroup}", tv_props.get_abs_prop_isolated],
        ]
        for name, func in time_variants:
            model.time_variants[name] = func
            model.parameters[name] = name

        # Tell the model to use these time varying functions for the stratification adjustments.
        agegroup_adj = flow_adjustments[f"within_{Compartment.EARLY_EXPOSED}Xagegroup_{agegroup}"]
        agegroup_adj[
            ClinicalStratum.SYMPT_NON_HOSPITAL
        ] = f"prop_{ClinicalStratum.SYMPT_NON_HOSPITAL}_{agegroup}"
        agegroup_adj[
            ClinicalStratum.SYMPT_ISOLATE
        ] = f"prop_{ClinicalStratum.SYMPT_ISOLATE}_{agegroup}"

    # Calculate death rates and progression rates for hospitalised and ICU patients
    progression_death_rates = {}
    for stratum in ("hospital", "icu"):
        (
            progression_death_rates[stratum + "_infect_death"],
            progression_death_rates[stratum + f"_within_{Compartment.LATE_ACTIVE}"],
        ) = find_rates_and_complements_from_ifr(
            rel_props[stratum + "_death"], 1, [model.parameters[f"within_{stratum}_late"]] * 16,
        )

    # Death and non-death progression between infectious compartments towards the recovered compartment
    for param in (f"within_{Compartment.LATE_ACTIVE}", "infect_death"):
        flow_adjustments.update(
            adjust_upstream_stratified_parameter(
                param,
                hospital_strata,
                "agegroup",
                agegroup_strata,
                [
                    progression_death_rates[f"hospital_{param}"],
                    progression_death_rates[f"icu_{param}"],
                ],
                overwrite=True,
            )
        )

    # Over-write rate of progression for early compartments for hospital and ICU
    # FIXME: Ask Romain if he knows why we bother doing this.
    flow_adjustments.update(
        {
            f"within_{Compartment.EARLY_ACTIVE}Xagegroup_{agegroup}": {
                f"{ClinicalStratum.HOSPITAL_NON_ICU}W": within_hospital_early,
                f"{ClinicalStratum.ICU}W": within_icu_early,
            }
            for agegroup in agegroup_strata
        }
    )

    # Make infectiousness adjustment for asymptomatic patients only
    strata_infectiousness = {}
    for stratum in clinical_strata:
        if stratum + "_infect_multiplier" in params:
            strata_infectiousness[stratum] = params[stratum + "_infect_multiplier"]

    # Make adjustment for isolation/quarantine
    # Allow pre-symptomatics to be less infectious
    clinical_inf_overwrites = [
        {
            "comp_name": Compartment.LATE_EXPOSED,
            "comp_strata": {},
            "value": params[f"{Compartment.LATE_EXPOSED}_infect_multiplier"],
        }
    ]

    for stratum in clinical_strata:
        if stratum in params["late_infect_multiplier"]:
            adjustment = {
                "comp_name": Compartment.LATE_ACTIVE,
                "comp_strata": {"clinical": stratum},
                "value": params["late_infect_multiplier"][stratum],
            }
            clinical_inf_overwrites.append(adjustment)

    # FIXME: This is not a desirable API, it's not really clear what is happening.
    model.individual_infectiousness_adjustments = clinical_inf_overwrites

    # FIXME: Ask Romain about importation
    # Work out time-variant clinical proportions for imported cases accounting for quarantine
    if implement_importation:
        to_infectious_key = f"within_{Compartment.EARLY_EXPOSED}Xagegroup_" + str(
            params["import_representative_age"]
        )
        tvs = model.time_variants

        # Create scale-up function for quarantine
        quarantine_scale_up = scale_up_function(
            traveller_quarantine["times"], traveller_quarantine["values"], method=4
        )

        # Clinical proportions for imported cases stay fixed for the hospitalised and critical groups (last two strata)
        importation_props_by_clinical = {
            stratum: float(flow_adjustments[to_infectious_key][stratum])
            for stratum in hospital_strata
        }

        # Proportion entering non-symptomatic stratum reduced by the quarantined (and so isolated) proportion
        def tv_prop_imported_non_sympt(t):
            return flow_adjustments[to_infectious_key][ClinicalStratum.NON_SYMPT] * (
                1.0 - quarantine_scale_up(t)
            )

        # Proportion ambulatory also reduced by quarantined proportion due to isolation
        def tv_prop_imported_sympt_non_hospital(t):
            return tvs[flow_adjustments[to_infectious_key][ClinicalStratum.SYMPT_NON_HOSPITAL]](
                t
            ) * (1.0 - quarantine_scale_up(t))

        # Proportion isolated includes those that would have been detected anyway and the ones above quarantined
        def tv_prop_imported_sympt_isolate(t):
            return tvs[flow_adjustments[to_infectious_key][ClinicalStratum.SYMPT_ISOLATE]](
                t
            ) + quarantine_scale_up(t) * (
                tvs[flow_adjustments[to_infectious_key][ClinicalStratum.SYMPT_NON_HOSPITAL]](t)
                + flow_adjustments[to_infectious_key][ClinicalStratum.NON_SYMPT]
            )

        # Pass time-variant functions to the model object
        model.time_variants[
            f"tv_prop_imported_{ClinicalStratum.NON_SYMPT}"
        ] = tv_prop_imported_non_sympt
        model.time_variants[
            f"tv_prop_imported_{ClinicalStratum.SYMPT_NON_HOSPITAL}"
        ] = tv_prop_imported_sympt_non_hospital
        model.time_variants[
            f"tv_prop_imported_{ClinicalStratum.SYMPT_ISOLATE}"
        ] = tv_prop_imported_sympt_isolate
        for stratum in non_hospital_strata:
            importation_props_by_clinical[stratum] = f"tv_prop_imported_{stratum}"

    else:
        importation_props_by_clinical = {}

    if params["implement_importation"]:
        for agegroup in agegroup_strata:
            key = f"importation_rateXagegroup_{agegroup}"
            flow_adjustments[key] = importation_props_by_clinical

    # Stratify the model using the SUMMER stratification function
    model.stratify(
        "clinical",
        clinical_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr)
    according to the split proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1 - split_props)
    return split_arr, complement_arr


class TimeVaryingProportions:
    """
    Provides time-varying proportions for a given age group.
    The proportions determine which clinical stratum people transition into when they go
    from being early exposed to late exposed.
    """

    def __init__(self, age_idx, abs_props, prop_detect_func):
        self.abs_prop_sympt = abs_props["sympt"][age_idx]
        self.abs_prop_hospital = abs_props["hospital"][age_idx]
        self.prop_detect_among_sympt_func = prop_detect_func

    def get_abs_prop_isolated(self, t):
        """
        Returns the absolute proportion of infected becoming isolated at home.
        Isolated people are those who are detected but not sent to hospital.
        """
        abs_prop_detected = self.abs_prop_sympt * self.prop_detect_among_sympt_func(t)
        abs_prop_isolated = abs_prop_detected - self.abs_prop_hospital
        if abs_prop_isolated < 0:
            # If more people go to hospital than are detected, ignore detection
            # proportion, and assume no one is being isolated.
            abs_prop_isolated = 0

        return abs_prop_isolated

    def get_abs_prop_sympt_non_hospital(self, t):
        """
        Returns the absolute proportion of infected not entering the hospital.
        This also does not count people who are isolated.
        This is only people who are not detected.
        """
        return self.abs_prop_sympt - self.abs_prop_hospital - self.get_abs_prop_isolated(t)
