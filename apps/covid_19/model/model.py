from types import MethodType

import numpy as np

from summer2 import CompartmentalModel, Stratification, Multiply, Overwrite

from autumn import inputs
from autumn.tool_kit.utils import apply_odds_ratio_to_proportion
from autumn.mixing.mixing import create_assortative_matrix
from autumn.curve import tanh_based_scaleup, scale_up_function
from autumn.environment.seasonality import get_seasonal_forcing
from autumn.tool_kit.utils import (
    normalise_sequence,
    repeat_list_elements,
    repeat_list_elements_average_last_two,
)

from apps.covid_19.constants import Compartment, Clinical as Clinical
from apps.covid_19.mixing_optimisation.constants import OPTI_ISO3S, Region
from apps.covid_19.model import outputs, preprocess
from apps.covid_19.model.importation import get_all_vic_notifications
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.preprocess.testing import find_cdr_function_from_test_data
from apps.covid_19.model.preprocess.case_detection import (
    build_detected_proportion_func,
    get_testing_pop,
)
from apps.covid_19.model.victorian_mixing import build_victorian_mixing_matrix_func
from apps.covid_19.model.victorian_outputs import add_victorian_derived_outputs

MOB_REGIONS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]

"""
Compartments
"""
# People who are infectious.
INFECTIOUS_COMPARTMENTS = [
    Compartment.LATE_EXPOSED,
    Compartment.EARLY_ACTIVE,
    Compartment.LATE_ACTIVE,
]
# People who are infected, but may or may not be infectuous.
DISEASE_COMPARTMENTS = [Compartment.EARLY_EXPOSED, *INFECTIOUS_COMPARTMENTS]
# All model compartments
COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED, *DISEASE_COMPARTMENTS]


def build_model(params: dict) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)
    model = CompartmentalModel(
        times=[params.time.start, params.time.end],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
        timestep=params.time.step,
    )

    """
    Population distribution
    """
    country = params.country
    pop = params.population
    agegroup_strata = [
        str(s)
        for s in range(
            0, params.age_stratification.max_age, params.age_stratification.age_step_size
        )
    ]
    # Time periods calculated from periods (ie "sojourn times")
    compartment_periods = preprocess.compartments.calc_compartment_periods(params.sojourn)

    # Get country population by age-group
    total_pops = inputs.get_population_by_agegroup(
        agegroup_strata, country.iso3, pop.region, year=pop.year
    )

    # Distribute infectious seed across infectious split sub-compartments
    total_disease_time = sum([compartment_periods[c] for c in DISEASE_COMPARTMENTS])
    init_pop = {
        c: params.infectious_seed * compartment_periods[c] / total_disease_time
        for c in DISEASE_COMPARTMENTS
    }

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows
    """
    # Infection flows with option to use seasonal forcing.
    if params.seasonal_force:
        # Use a time-varying, sinusoidal seasonal forcing function for contact rate.
        contact_rate = get_seasonal_forcing(
            365.0, 173.0, params.seasonal_force, params.contact_rate
        )
    else:
        # Use a static contact rate.
        contact_rate = params.contact_rate

    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.EARLY_EXPOSED,
    )
    # Infection progress flows.
    model.add_sojourn_flow(
        name="infect_onset",
        sojourn_time=compartment_periods[Compartment.EARLY_EXPOSED],
        source=Compartment.EARLY_EXPOSED,
        dest=Compartment.LATE_EXPOSED,
    )
    model.add_sojourn_flow(
        name="incidence",
        sojourn_time=compartment_periods[Compartment.LATE_EXPOSED],
        source=Compartment.LATE_EXPOSED,
        dest=Compartment.EARLY_ACTIVE,
    )
    model.add_sojourn_flow(
        name="progress",
        sojourn_time=compartment_periods[Compartment.EARLY_ACTIVE],
        source=Compartment.EARLY_ACTIVE,
        dest=Compartment.LATE_ACTIVE,
    )
    # Recovery flows
    model.add_sojourn_flow(
        name="recovery",
        sojourn_time=compartment_periods[Compartment.LATE_ACTIVE],
        source=Compartment.LATE_ACTIVE,
        dest=Compartment.RECOVERED,
    )
    # Infection death
    model.add_death_flow(
        name="infect_death",
        death_rate=0,  # Will be overwritten later in clinical stratification.
        source=Compartment.LATE_ACTIVE,
    )

    if params.waning_immunity_duration is not None:
        # Waning immunity (if requested)
        model.add_sojourn_flow(
            name="warning_immunity",
            sojourn_time=params.waning_immunity_duration,
            source=Compartment.RECOVERED,
            dest=Compartment.SUSCEPTIBLE,
        )

    """
    Optionally add an importation flow, where we ship in infected people from overseas.
    """
    importation = params.importation
    # Get proportion of importations by age. This is used to calculate case detection and used in age stratification.
    importation_props_by_age = (
        importation.props_by_age
        if importation and importation.props_by_age
        else {s: 1.0 / len(agegroup_strata) for s in agegroup_strata}
    )
    # Get case detection rate function.
    get_detected_proportion = build_detected_proportion_func(
        agegroup_strata, country, pop, params.testing_to_detection, params.case_detection
    )
    # Determine how many importations there are, including the undetected and asymptomatic importations
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    symptomatic_props = repeat_list_elements(
        2, params.clinical_stratification.props.symptomatic.props
    )
    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(importation_props_by_age.values(), symptomatic_props)
        ]
    )

    def modelled_abs_detection_proportion_imported(t):
        # Returns absolute proprotion of imported people who are detected to be infectious.
        return import_symptomatic_prop * get_detected_proportion(t)

    if importation:
        is_region_vic = pop.region and Region.to_name(pop.region) in Region.VICTORIA_SUBREGIONS
        if is_region_vic:
            import_times, importation_data = get_all_vic_notifications(
                excluded_regions=(pop.region,)
            )
            testing_pop, _ = get_testing_pop(agegroup_strata, country, pop)
            movement_to_region = (
                sum(total_pops) / sum(testing_pop) * params.importation.movement_prop
            )
            import_cases = [i_cases * movement_to_region for i_cases in importation_data]
        else:
            import_times = params.importation.case_timeseries.times
            import_cases = params.importation.case_timeseries.values

        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times, import_cases, modelled_abs_detection_proportion_imported
        )

        # Imported people are infectious (ie. late active).
        model.add_importation_flow(
            name="importation",
            num_imported=import_rate_func,
            dest=Compartment.LATE_ACTIVE,
        )

    """
    Age stratification
    """
    # We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    # automatic demography features (which work on the assumption that the time unit is years, so would be totally wrong)
    age_strat = Stratification("agegroup", agegroup_strata, COMPARTMENTS)

    # Dynamic heterogeneous mixing by age
    if params.elderly_mixing_reduction and not params.mobility.age_mixing:
        # Apply eldery protection to the age mixing parameters
        params.mobility.age_mixing = preprocess.elderly_protection.get_elderly_protection_mixing(
            params.elderly_mixing_reduction
        )

    static_mixing_matrix = inputs.get_country_mixing_matrix("all_locations", country.iso3)
    dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic_mixing_matrix(
        static_mixing_matrix,
        params.mobility,
        country,
    )
    age_strat.set_mixing_matrix(dynamic_mixing_matrix)

    # Set distribution of starting population
    age_split_props = {
        agegroup: prop for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }
    age_strat.set_population_split(age_split_props)

    # Adjust flows based on age group.
    age_strat.add_flow_adjustments(
        "infection", {s: Multiply(v) for s, v in params.age_stratification.susceptibility.items()}
    )
    if importation:
        age_strat.add_flow_adjustments(
            "importation", {s: Multiply(v) for s, v in importation_props_by_age.items()}
        )

    model.stratify_with(age_strat)

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
    clinical_strata = [
        Clinical.NON_SYMPT,
        Clinical.SYMPT_NON_HOSPITAL,
        Clinical.SYMPT_ISOLATE,
        Clinical.HOSPITAL_NON_ICU,
        Clinical.ICU,
    ]
    non_hospital_strata = [
        Clinical.NON_SYMPT,
        Clinical.SYMPT_NON_HOSPITAL,
        Clinical.SYMPT_ISOLATE,
    ]
    hospital_strata = [
        Clinical.HOSPITAL_NON_ICU,
        Clinical.ICU,
    ]
    clinical_strat = Stratification("clinical", clinical_strata, INFECTIOUS_COMPARTMENTS)
    clinical_params = params.clinical_stratification

    """
    Infectiousness adjustments for clinical strat
    """
    # Some euro models contain the assumption that late exposed people are less infectious.
    # For most models this is a null-op with the infectiousness adjustment being 1.
    adjust = Overwrite(clinical_params.late_exposed_infect_multiplier)
    clinical_strat.add_infectiousness_adjustments(
        Compartment.LATE_EXPOSED,
        {
            Clinical.NON_SYMPT: adjust,
            Clinical.SYMPT_ISOLATE: adjust,
            Clinical.SYMPT_NON_HOSPITAL: adjust,
            Clinical.HOSPITAL_NON_ICU: adjust,
            Clinical.ICU: adjust,
        },
    )

    # Add infectiousness reduction multiplier for all non-symptomatic infectious people.
    # These people are less infectious because of biology.
    for comp in INFECTIOUS_COMPARTMENTS:
        clinical_strat.add_infectiousness_adjustments(
            comp,
            {
                Clinical.NON_SYMPT: Overwrite(clinical_params.non_sympt_infect_multiplier),
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
            Clinical.NON_SYMPT: None,
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
    hospital_death_rates = (
        relative_death_props[Clinical.HOSPITAL_NON_ICU] * model.parameters[f"within_hospital_late"]
    )
    icu_death_rates = relative_death_props[Clinical.ICU] * model.parameters[f"within_icu_late"]

    # Apply adjusted infection death rates for hospital patients (ICU and non-ICU)
    # Death and non-death progression between infectious compartments towards the recovered compartment
    for idx, agegroup in enumerate(agegroup_strata):
        clinical_strat.add_flow_adjustments(
            "infect_death",
            {
                Clinical.NON_SYMPT: None,
                Clinical.SYMPT_NON_HOSPITAL: None,
                Clinical.SYMPT_ISOLATE: None,
                Clinical.HOSPITAL_NON_ICU: Overwrite(hospital_death_rates[idx]),
                Clinical.ICU: Overwrite(icu_death_rates[idx]),
            },
            source_strata={"agegroup": agegroup},
        )

    """
    Adjust early exposed sojourn times.
    """
    # Progression rates into the infectious compartment(s)
    # Define progression rates into non-symptomatic compartments using parameter adjustment.
    for age_idx, agegroup in enumerate(agegroup_strata):
        get_abs_prop_isolated = get_abs_prop_isolated_factory(
            age_idx, abs_props, get_detected_proportion
        )
        get_abs_prop_sympt_non_hospital = get_abs_prop_sympt_non_hospital_factory(
            age_idx, abs_props, get_abs_prop_isolated
        )
        clinical_strat.add_flow_adjustments(
            "infect_onset",
            {
                Clinical.NON_SYMPT: Multiply(abs_props[Clinical.NON_SYMPT][age_idx]),
                Clinical.ICU: Multiply(abs_props[Clinical.ICU][age_idx]),
                Clinical.HOSPITAL_NON_ICU: Multiply(abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]),
                Clinical.SYMPT_NON_HOSPITAL: Multiply(get_abs_prop_sympt_non_hospital),
                Clinical.SYMPT_ISOLATE: Multiply(get_abs_prop_isolated),
            },
            source_strata={"agegroup": agegroup},
        )

    """
    Adjust early active sojourn times.
    """
    # Over-write rate of progression for early compartments for hospital and ICU
    within_hospital_early = model.parameters["within_hospital_early"]
    within_icu_early = model.parameters["within_icu_early"]
    for agegroup in agegroup_strata:
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

    hospital_survival_rates = hospital_survival_props * model.parameters[f"within_hospital_late"]
    icu_survival_rates = icu_survival_props * model.parameters[f"within_icu_late"]

    late_active_adjs = {}
    for idx, agegroup in enumerate(agegroup_strata):
        age_key = f"agegroup_{agegroup}"
        late_active_adjs[f"within_{Compartment.LATE_ACTIVE}X{age_key}"] = {
            "hospital_non_icuW": hospital_survival_rates[idx],
            "icuW": icu_survival_rates[idx],
        }

    """
    Clinical proportions for imported cases.
    """

    # Work out time-variant clinical proportions for imported cases accounting for quarantine
    import_adjs = {}
    if importation is not None:
        tvs = model.time_variants
        importation_props_by_clinical = {}

        # Create scale-up function for quarantine
        quarantine_timeseries = importation.quarantine_timeseries
        if quarantine_timeseries.times:
            # Construct a quarantine function from timeseries.
            quarantine_func = scale_up_function(
                quarantine_timeseries.times, quarantine_timeseries.values, method=4
            )
        else:
            # Default to no quarantine if not values are available.
            quarantine_func = lambda _: 0

        # Loop through age groups and set the appropriate clinical proportions
        for agegroup in agegroup_strata:
            param_key = f"within_{Compartment.EARLY_EXPOSED}Xagegroup_{agegroup}"

            # Proportion entering non-symptomatic stratum reduced by the quarantined (and so isolated) proportion
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{Clinical.NON_SYMPT}"
            ] = lambda t: early_exposed_adjs[param_key][Clinical.NON_SYMPT] * (
                1.0 - quarantine_func(t)
            )

            # Proportion ambulatory also reduced by quarantined proportion due to isolation
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{Clinical.SYMPT_NON_HOSPITAL}"
            ] = lambda t: tvs[early_exposed_adjs[param_key][Clinical.SYMPT_NON_HOSPITAL]](t) * (
                1.0 - quarantine_func(t)
            )

            # Proportion isolated includes those that would have been detected anyway and the ones above quarantined
            model.time_variants[
                f"tv_prop_importedX{agegroup}X{Clinical.SYMPT_ISOLATE}"
            ] = lambda t: quarantine_func(t) * (
                tvs[early_exposed_adjs[param_key][Clinical.SYMPT_NON_HOSPITAL]](t)
                + early_exposed_adjs[param_key][Clinical.NON_SYMPT]
            ) + tvs[
                early_exposed_adjs[param_key][Clinical.SYMPT_ISOLATE]
            ](
                t
            )

            # Create request with correct syntax for SUMMER for hospitalised and then non-hospitalised
            importation_props_by_clinical[agegroup] = {
                stratum: float(early_exposed_adjs[param_key][stratum])
                for stratum in hospital_strata
            }
            importation_props_by_clinical[agegroup].update(
                {
                    stratum: f"tv_prop_importedX{agegroup}X{stratum}"
                    for stratum in non_hospital_strata
                }
            )
            import_adjs[f"importation_rateXagegroup_{agegroup}"] = importation_props_by_clinical[
                agegroup
            ]

    # Only stratify infected compartments
    compartments_to_stratify = [
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
    ]

    flow_adjustments = {
        **early_exposed_adjs,
        **early_active_adjs,
        **late_active_adjs,
        **import_adjs,
    }

    # Stratify the model using the SUMMER stratification function
    model.stratify(
        "clinical",
        clinical_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )

    """
    Infection history stratification
    """
    if params.stratify_by_infection_history:
        # waning immunity makes recovered individuals transition to the 'experienced' stratum
        stratification_adjustments = {"immunity_loss_rate": {"naive": 0.0, "experienced": 1.0}}
        # adjust parameters defining progression from early exposed to late exposed to obtain the requested proportion
        for agegroup in agegroup_strata:  # e.g. '0'
            # collect existing rates of progressions for symptomatic vs non-symptomatic
            rate_non_sympt = (
                early_exposed_adjs[f"within_early_exposedXagegroup_{agegroup}"]["non_sympt"]
                * model.parameters["within_early_exposed"]
            )
            total_progression_rate = model.parameters["within_early_exposed"]
            rate_sympt = total_progression_rate - rate_non_sympt

            # multiplier for symptomatic is rel_prop_symptomatic_experienced
            sympt_multiplier = params.rel_prop_symptomatic_experienced
            # multiplier for asymptomatic rate is 1 + rate_sympt / rate_non_sympt * (1 - sympt_multiplier)
            # in order to preserve aggregated exit flow
            non_sympt_multiplier = 1 + rate_sympt / rate_non_sympt * (1.0 - sympt_multiplier)

            # create adjustment requests
            for clinical_stratum in clinical_strata:
                param_name = (
                    f"within_early_exposedXagegroup_{agegroup}Xclinical_" + clinical_stratum
                )
                if clinical_stratum == "non_sympt":
                    stratification_adjustments[param_name] = {
                        "naive": 1.0,
                        "experienced": non_sympt_multiplier,
                    }
                else:
                    stratification_adjustments[param_name] = {
                        "naive": 1.0,
                        "experienced": sympt_multiplier,
                    }

        # Stratify the model using the SUMMER stratification function
        model.stratify(
            "history",
            ["naive", "experienced"],
            [Compartment.SUSCEPTIBLE, Compartment.EARLY_EXPOSED],
            comp_split_props={"naive": 1.0, "experienced": 0.0},
            flow_adjustments=stratification_adjustments,
        )

    """
    Stratify model by Victorian subregion (used for Victorian cluster model).
    """
    if params.victorian_clusters:
        vic = params.victorian_clusters
        # Determine how to split up population by cluster
        # There is -0.5% to +4% difference per age group between sum of region population in 2018 and
        # total VIC population in 2020
        cluster_strata = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
        region_pops = {
            region: sum(
                inputs.get_population_by_agegroup(
                    agegroup_strata, country.iso3, region.upper(), year=2018
                )
            )
            for region in cluster_strata
        }
        sum_region_props = sum(region_pops.values())
        cluster_split_props = {
            region: pop / sum_region_props for region, pop in region_pops.items()
        }

        # Adjust contact rate multipliers
        contact_rate_multipliers = {}
        for cluster in Region.VICTORIA_SUBREGIONS:
            cluster_name = cluster.replace("-", "_")
            adjustment = eval(f"vic.contact_rate_multiplier_{cluster_name}")
            contact_rate_multipliers.update({cluster_name: adjustment})

        # Add in flow adjustments per-region so we can calibrate the contact rate for each region.
        cluster_flow_adjustments = {}
        for agegroup_stratum in agegroup_strata:
            param_name = f"contact_rateXagegroup_{agegroup_stratum}"
            cluster_flow_adjustments[param_name] = contact_rate_multipliers

        # Use an identity mixing matrix to temporarily declare no inter-cluster mixing, which will then be over-written
        cluster_mixing_matrix = np.eye(len(cluster_strata))

        model.stratify(
            "cluster",
            cluster_strata,
            COMPARTMENTS,
            comp_split_props=cluster_split_props,
            flow_adjustments=cluster_flow_adjustments,
            mixing_matrix=cluster_mixing_matrix,
        )

        regional_clusters = [region.replace("-", "_") for region in Region.VICTORIA_RURAL]

        # A bit of a hack - to get rid of all the infectious populations from the regional clusters
        for i_comp, comp in enumerate(model.compartment_names):
            if any(
                [comp.has_stratum("cluster", cluster) for cluster in regional_clusters]
            ) and not comp.has_name(Compartment.SUSCEPTIBLE):
                model.compartment_values[i_comp] = 0.0

        """
        Hack in a custom (144x144) mixing matrix where each region is adjusted individually
        based on its time variant mobility data.
        """

        # Get the inter-cluster mixing matrix
        intercluster_mixing_matrix = create_assortative_matrix(vic.intercluster_mixing, MOB_REGIONS)

        # Replace regional Victoria maximum effect calibration parameters with the metro values for consistency
        for param_to_copy in ["face_coverings", "behaviour"]:
            vic.regional.mobility.microdistancing[
                param_to_copy
            ].parameters.upper_asymptote = vic.metro.mobility.microdistancing[
                param_to_copy
            ].parameters.upper_asymptote

        # Get new mixing matrix
        get_mixing_matrix = build_victorian_mixing_matrix_func(
            static_mixing_matrix,
            vic.metro.mobility,
            vic.regional.mobility,
            country,
            intercluster_mixing_matrix,
        )
        setattr(model, "get_mixing_matrix", MethodType(get_mixing_matrix, model))

    """
    Set up and track derived output functions
    """
    if not params.victorian_clusters:
        # Set up derived outputs
        incidence_connections = outputs.get_incidence_connections(model.compartment_names)
        progress_connections = outputs.get_progress_connections(model.compartment_names)
        death_output_connections = outputs.get_infection_death_connections(model.compartment_names)
        model.add_flow_derived_outputs(incidence_connections)
        model.add_flow_derived_outputs(progress_connections)
        model.add_flow_derived_outputs(death_output_connections)

        # Build notification derived output function
        is_importation_active = params.importation is not None
        notification_func = outputs.get_calc_notifications_covid(
            is_importation_active,
            modelled_abs_detection_proportion_imported,
        )
        local_notification_func = outputs.get_calc_notifications_covid(
            False, modelled_abs_detection_proportion_imported
        )

        # Build life expectancy derived output function
        life_expectancy = inputs.get_life_expectancy_by_agegroup(agegroup_strata, country.iso3)[0]
        life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]
        life_lost_func = outputs.get_calculate_years_of_life_lost(life_expectancy_latest)

        # Build hospital occupancy func
        compartment_periods = params.sojourn.compartment_periods
        icu_early_period = compartment_periods["icu_early"]
        hospital_early_period = compartment_periods["hospital_early"]
        calculate_hospital_occupancy = outputs.get_calculate_hospital_occupancy(
            model, icu_early_period, hospital_early_period
        )

        func_outputs = {
            # Case-related
            "notifications": notification_func,
            "local_notifications": local_notification_func,
            "notifications_at_sympt_onset": outputs.get_notifications_at_sympt_onset,
            # Death-related
            "years_of_life_lost": life_lost_func,
            "accum_deaths": outputs.calculate_cum_deaths,
            # Health care-related
            "hospital_occupancy": calculate_hospital_occupancy,
            "icu_occupancy": outputs.calculate_icu_occupancy,
            "new_hospital_admissions": outputs.calculate_new_hospital_admissions_covid,
            "new_icu_admissions": outputs.calculate_new_icu_admissions_covid,
            # Other
            "proportion_seropositive": outputs.calculate_proportion_seropositive,
        }

        # Derived outputs for the optimization project.
        if params.country.iso3 in OPTI_ISO3S:
            func_outputs["accum_years_of_life_lost"] = outputs.calculate_cum_years_of_life_lost

        # Add age-specific proportion recovered for all applications except VIC clusters
        is_region_vic = pop.region and Region.to_name(pop.region) in Region.VICTORIA_SUBREGIONS
        if not is_region_vic:
            for agegroup in agegroup_strata:
                age_key = f"agegroup_{agegroup}"
                func_outputs[
                    f"proportion_seropositiveX{age_key}"
                ] = outputs.make_age_specific_seroprevalence_output(agegroup)
                func_outputs[f"accum_deathsX{age_key}"] = outputs.make_agespecific_cum_deaths_func(
                    agegroup
                )

        model.add_function_derived_outputs(func_outputs)
    else:
        add_victorian_derived_outputs(
            model,
            icu_early_period=compartment_periods["icu_early"],
            hospital_early_period=compartment_periods["hospital_early"],
        )

    return model


def get_abs_prop_isolated_factory(age_idx, abs_props, prop_detect_among_sympt_func):
    def get_abs_prop_isolated(t):
        """
        Returns the absolute proportion of infected becoming isolated at home.
        Isolated people are those who are detected but not sent to hospital.
        """
        abs_prop_detected = abs_props["sympt"][age_idx] * prop_detect_among_sympt_func(t)
        abs_prop_isolated = abs_prop_detected - abs_props["hospital"][age_idx]
        if abs_prop_isolated < 0:
            # If more people go to hospital than are detected, ignore detection
            # proportion, and assume no one is being isolated.
            abs_prop_isolated = 0

        return abs_prop_isolated

    return get_abs_prop_isolated


def get_abs_prop_sympt_non_hospital_factory(age_idx, abs_props, get_abs_prop_isolated_func):
    def get_abs_prop_sympt_non_hospital(t):
        """
        Returns the absolute proportion of infected not entering the hospital.
        This also does not count people who are isolated.
        This is only people who are not detected.
        """
        return (
            abs_props["sympt"][age_idx]
            - abs_props["hospital"][age_idx]
            - get_abs_prop_isolated_func(t)
        )

    return get_abs_prop_sympt_non_hospital


def get_absolute_strata_proportions(
    symptomatic_props,
    icu_props,
    hospital_props,
    symptomatic_props_multiplier,
    hospital_props_multiplier,
):
    """
    Returns the proportion of people in each clinical stratum.
    ie: Given all the people people who are infected, what proportion are in each strata?
    Each of these are stratified into 16 age groups 0-75+
    """
    # Apply multiplier to proportions
    hospital_props = [
        apply_odds_ratio_to_proportion(i_prop, hospital_props_multiplier)
        for i_prop in hospital_props
    ]
    symptomatic_props = [
        apply_odds_ratio_to_proportion(i_prop, symptomatic_props_multiplier)
        for i_prop in symptomatic_props
    ]

    # Find the absolute progression proportions.
    symptomatic_props_arr = np.array(symptomatic_props)
    hospital_props_arr = np.array(hospital_props)
    # Determine the absolute proportion of early exposed who become sympt vs non-sympt
    sympt, non_sympt = subdivide_props(1, symptomatic_props_arr)
    # Determine the absolute proportion of sympt who become hospitalized vs non-hospitalized.
    sympt_hospital, sympt_non_hospital = subdivide_props(sympt, hospital_props_arr)
    # Determine the absolute proportion of hospitalized who become icu vs non-icu.
    sympt_hospital_icu, sympt_hospital_non_icu = subdivide_props(sympt_hospital, icu_props)

    return {
        "sympt": sympt,
        "non_sympt": non_sympt,
        "hospital": sympt_hospital,
        "sympt_non_hospital": sympt_non_hospital,  # Over-ridden by a by time-varying proportion later
        Clinical.ICU: sympt_hospital_icu,
        Clinical.HOSPITAL_NON_ICU: sympt_hospital_non_icu,
    }


def get_absolute_death_proportions(abs_props, infection_fatality_props, icu_mortality_prop):
    """
    Calculate death proportions: find where the absolute number of deaths accrue
    Represents the number of people in a strata who die given the total number of people infected.
    """
    NUM_AGE_STRATA = 16
    abs_death_props = {
        Clinical.NON_SYMPT: np.zeros(NUM_AGE_STRATA),
        Clinical.ICU: np.zeros(NUM_AGE_STRATA),
        Clinical.HOSPITAL_NON_ICU: np.zeros(NUM_AGE_STRATA),
    }
    for age_idx in range(NUM_AGE_STRATA):
        age_ifr_props = infection_fatality_props[age_idx]

        # Make sure there are enough asymptomatic and hospitalised proportions to fill the IFR
        thing = (
            abs_props["non_sympt"][age_idx]
            + abs_props[Clinical.HOSPITAL_NON_ICU][age_idx]
            + abs_props[Clinical.ICU][age_idx] * icu_mortality_prop
        )
        age_ifr_props = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in ICU
        # Maximum ICU mortality allowed
        thing = abs_props[Clinical.ICU][age_idx] * icu_mortality_prop
        abs_death_props[Clinical.ICU][age_idx] = min(thing, age_ifr_props)

        # Absolute proportion of all patients dying in hospital, excluding ICU
        thing = max(
            age_ifr_props
            - abs_death_props[Clinical.ICU][
                age_idx
            ],  # If left over mortality from ICU for hospitalised
            0.0,  # Otherwise zero
        )
        abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx] = min(
            thing,
            # Otherwise fill up hospitalised
            abs_props[Clinical.HOSPITAL_NON_ICU][age_idx],
        )

        # Absolute proportion of all patients dying out of hospital
        thing = (
            age_ifr_props
            - abs_death_props[Clinical.ICU][age_idx]
            - abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
        )  # If left over mortality from hospitalised
        abs_death_props[Clinical.NON_SYMPT][age_idx] = max(0.0, thing)  # Otherwise zero

        # Check everything sums up properly
        allowed_rounding_error = 6
        assert (
            round(
                abs_death_props[Clinical.ICU][age_idx]
                + abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
                + abs_death_props[Clinical.NON_SYMPT][age_idx],
                allowed_rounding_error,
            )
            == round(age_ifr_props, allowed_rounding_error)
        )
        # Check everything sums up properly
        allowed_rounding_error = 6
        assert (
            round(
                abs_death_props[Clinical.ICU][age_idx]
                + abs_death_props[Clinical.HOSPITAL_NON_ICU][age_idx]
                + abs_death_props[Clinical.NON_SYMPT][age_idx],
                allowed_rounding_error,
            )
            == round(age_ifr_props, allowed_rounding_error)
        )

    return abs_death_props


def get_infection_fatality_proportions(
    infection_fatality_props_10_year,
    infection_rate_multiplier,
    iso3,
    pop_region,
    pop_year,
):
    """
    Returns the Proportion of people in age group who die, given the total number of people in that compartment.
    ie: dead / total infected
    """
    if_props_10_year = [
        apply_odds_ratio_to_proportion(i_prop, infection_rate_multiplier)
        for i_prop in infection_fatality_props_10_year
    ]
    # Calculate the proportion of 80+ years old among the 75+ population
    elderly_populations = inputs.get_population_by_agegroup(
        [0, 75, 80], iso3, pop_region, year=pop_year
    )
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])
    # Infection fatality rate by age group.
    # Data in props may have used 10 year bands 0-80+, but we want 5 year bands from 0-75+
    # Calculate 75+ age bracket as weighted average between 75-79 and half 80+
    if len(infection_fatality_props_10_year) == 17:
        last_ifr = if_props_10_year[-1] * prop_over_80 + if_props_10_year[-2] * (1 - prop_over_80)
        ifrs_by_age = if_props_10_year[:-1]
        ifrs_by_age[-1] = last_ifr
    else:
        ifrs_by_age = repeat_list_elements_average_last_two(if_props_10_year, prop_over_80)
    return ifrs_by_age


def subdivide_props(base_props: np.ndarray, split_props: np.ndarray):
    """
    Split an array (base_array) of proportions into two arrays (split_arr, complement_arr)
    according to the split proportions provided (split_prop).
    """
    split_arr = base_props * split_props
    complement_arr = base_props * (1 - split_props)
    return split_arr, complement_arr
