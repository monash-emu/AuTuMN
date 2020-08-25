import os
from copy import deepcopy

import numpy as np

from autumn import inputs
from autumn.constants import Flow, BirthApproach
from autumn.curve import tanh_based_scaleup, scale_up_function
from autumn.environment.seasonality import get_seasonal_forcing
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.utils import normalise_sequence, repeat_list_elements
from summer.model import StratifiedModel

from data.inputs.testing.testing import get_vic_testing_numbers

from apps.covid_19.constants import Compartment, ClinicalStratum
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS

from . import outputs, preprocess
from .stratification import stratify_by_clinical
from .preprocess.testing import create_cdr_function
from .validate import validate_params


def build_model(params: dict) -> StratifiedModel:
    """
    Build the compartmental model from the provided parameters.
    """
    validate_params(params)

    # Get country population by age-group.
    agegroup_max, agegroup_step = params["agegroup_breaks"]
    agegroup_strata = list(range(0, agegroup_max, agegroup_step))
    country_iso3 = params["iso3"]
    region = params["region"]
    pop_region = (
        params["pop_region_override"] if params["pop_region_override"] else params["region"]
    )
    total_pops = inputs.get_population_by_agegroup(
        agegroup_strata, country_iso3, pop_region, year=params["pop_year"]
    )

    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_EXPOSED,
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
        Compartment.RECOVERED,
    ]

    # Define infectious, disease compartments.
    infectious_comps = [
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
    ]
    disease_comps = [Compartment.EARLY_EXPOSED, *infectious_comps]

    # Calculate time periods spent in various compartments.
    base_periods = params["compartment_periods"]
    periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_periods, periods_calc
    )

    # Distribute infectious seed across infectious compartments
    infectious_seed = params["infectious_seed"]
    total_disease_time = sum([compartment_periods[c] for c in disease_comps])
    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_disease_time for c in disease_comps
    }

    # Force the remainder starting population to go to S compartment (Required as entry_compartment is late_infectious)
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

    # Set integration times
    start_time = params["start_time"]
    end_time = params["end_time"]
    time_step = params["time_step"]
    times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    # Add inter-compartmental transition flows
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)
    flow_params = {
        "contact_rate": params["contact_rate"],
        "infect_death": 0,  # Overwritten in clinical stratification.
        # within_{comp_name} transition params are set below.
    }
    # Add parameters for the in-disease progression flows
    for comp_name, comp_period in compartment_periods.items():
        flow_params[f"within_{comp_name}"] = 1.0 / comp_period

    if not params["full_immunity"]:
        # Implement waning immunity
        flow_params["immunity_loss_rate"] = 1.0 / params["immunity_duration"]
        flows.append(
            {
                "type": Flow.STANDARD,
                "origin": Compartment.RECOVERED,
                "to": Compartment.SUSCEPTIBLE,
                "parameter": "immunity_loss_rate",
            }
        )

    implement_importation = params["implement_importation"]
    if implement_importation:
        # Implement importation of people, importation_rate is later as time varying function.
        flows.append({"type": Flow.IMPORT, "parameter": "importation_rate"})

    # Create SUMMER model
    model = StratifiedModel(
        times,
        compartments,
        init_pop,
        flow_params,
        flows,
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.LATE_ACTIVE,  # to model imported cases
        starting_population=sum(total_pops),
        infectious_compartments=infectious_comps,
    )

    # Build a dynamic, age-based mixing matrix.
    static_mixing_matrix = preprocess.mixing_matrix.build_static(country_iso3)
    dynamic_mixing_matrix = None
    dynamic_location_mixing_params = params["mixing"]
    dynamic_age_mixing_params = params["mixing_age_adjust"]
    microdistancing = params["microdistancing"]
    smooth_google_data = params["smooth_google_data"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    # FIXME: Why wouldn't we always use Google mobiliy data?
    if dynamic_location_mixing_params or dynamic_age_mixing_params:
        dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
            country_iso3,
            region,
            dynamic_location_mixing_params,
            dynamic_age_mixing_params,
            npi_effectiveness_params,
            google_mobility_locations,
            microdistancing,
            smooth_google_data,
        )
        model.set_dynamic_mixing_matrix(dynamic_mixing_matrix)

    # Implement seasonal forcing if requested, making contact rate a time-variant rather than constant
    if params["seasonal_force"]:
        seasonal_func = get_seasonal_forcing(
            365.0, 173.0, params["seasonal_force"], params["contact_rate"]
        )
        model.time_variants["contact_rate"] = seasonal_func

    # Stratify the model by age group.
    # Adjust flow parameters for different age strata.
    flow_adjustments = {
        # Adjust susceptibility across age groups
        "contact_rate": params["age_based_susceptibility"],
        # Adjust importation proportions
        "importation_rate": params["importation_props_by_age"],
    }
    # Distribute starting population over the different agegroups
    comp_split_props = {
        str(agegroup): prop
        for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }
    # We use "agegroup" instead of "age" for this model, to avoid triggering automatic demography features
    # (which work on the assumption that the time unit is years, so would be totally wrong)
    model.stratify(
        "agegroup",
        agegroup_strata,
        compartments_to_stratify=compartments,
        comp_split_props=comp_split_props,
        flow_adjustments=flow_adjustments,
        mixing_matrix=static_mixing_matrix,
    )

    # Determine the proportion of cases detected over time as `detected_proportion`.
    if params["testing_to_detection"]:

        # Parameters that will need to go into ymls
        assumed_tests_parameter = 1000.0
        assumed_cdr_parameter = 0.25

        # Tests numbers
        # FIXME: this should be made more general to any application
        test_dates, test_values = get_vic_testing_numbers()
        per_capita_tests = [i_tests / sum(total_pops) for i_tests in test_values]

        # Calculate CDRs and the resulting CDR function over time
        cdr_from_tests_func = create_cdr_function(assumed_tests_parameter, assumed_cdr_parameter)
        detected_proportion = scale_up_function(
            test_dates,
            [cdr_from_tests_func(i_tests) for i_tests in per_capita_tests],
            smoothness=0.2,
            method=5,
        )

    else:
        detect_prop_params = params["time_variant_detection"]

        # Create function describing the proportion of cases detected over time
        def detected_proportion(t):
            # Function representing the proportion of symptomatic people detected over time
            base_prop_detect = tanh_based_scaleup(
                detect_prop_params["maximum_gradient"],
                detect_prop_params["max_change_time"],
                detect_prop_params["start_value"],
                detect_prop_params["end_value"],
            )
            # Return value modified for any future intervention that narrows the case detection gap
            int_detect_gap_reduction = params["int_detection_gap_reduction"]
            return base_prop_detect(t) + (1.0 - base_prop_detect(t)) * int_detect_gap_reduction

    # Determine how many importations there are, including the undetected and asymptomatic importations
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    symptomatic_props = repeat_list_elements(2, params["symptomatic_props"])
    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(
                params["importation_props_by_age"].values(), symptomatic_props
            )
        ]
    )

    def modelled_abs_detection_proportion_imported(t):
        return import_symptomatic_prop * detected_proportion(t)

    # Set time-variant importation rate for the importation flow.
    if implement_importation:
        import_times = params["data"]["times_imported_cases"]
        import_cases = params["data"]["n_imported_cases"]
        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times, import_cases, modelled_abs_detection_proportion_imported
        )
        model.time_variants["importation_rate"] = import_rate_func

    # Stratify the model by clinical status.
    stratify_by_clinical(model, params, compartments, detected_proportion, symptomatic_props)
    # Finished building the model.

    # Set up derived outputs.
    # Define which flows we should track for derived outputs
    incidence_connections = outputs.get_incidence_connections(model.compartment_names)
    progress_connections = outputs.get_progress_connections(model.compartment_names)
    death_output_connections = outputs.get_infection_death_connections(model.compartment_names)
    model.add_flow_derived_outputs(incidence_connections)
    model.add_flow_derived_outputs(progress_connections)
    model.add_flow_derived_outputs(death_output_connections)

    # Build notification derived output function
    notification_func = outputs.get_calc_notifications_covid(
        implement_importation, modelled_abs_detection_proportion_imported,
    )
    local_notification_func = outputs.get_calc_notifications_covid(
        False, modelled_abs_detection_proportion_imported
    )

    # Build life expectancy derived output function.
    life_expectancy = inputs.get_life_expectancy_by_agegroup(agegroup_strata, country_iso3)[0]
    life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]
    life_lost_func = outputs.get_calculate_years_of_life_lost(life_expectancy_latest)

    # Build hospital occupancy func.
    compartment_periods = params["compartment_periods"]
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    calculate_hospital_occupancy = outputs.get_calculate_hospital_occupancy(
        icu_early_period, hospital_early_period
    )

    # Register derived output functions
    func_outputs = {
        "notifications": notification_func,
        "local_notifications": local_notification_func,
        "years_of_life_lost": life_lost_func,
        "prevXlate_activeXclinical_icuXamong": outputs.calculate_icu_prev,
        "hospital_occupancy": calculate_hospital_occupancy,
        "proportion_seropositive": outputs.calculate_proportion_seropositive,
        "new_hospital_admissions": outputs.calculate_new_hospital_admissions_covid,
        "proportion_seropositive": outputs.calculate_proportion_seropositive,
        "new_icu_admissions": outputs.calculate_new_icu_admissions_covid,
        "icu_occupancy": outputs.calculate_icu_occupancy,
        "notifications_at_sympt_onset": outputs.get_notifications_at_sympt_onset,
        "total_infection_deaths": outputs.get_infection_deaths,
    }
    if region in OPTI_REGIONS:
        # Derived outputs for the optimization project.
        func_outputs["accum_deaths"] = outputs.calculate_cum_deaths
        func_outputs["accum_years_of_life_lost"] = outputs.calculate_cum_years_of_life_lost

    model.add_function_derived_outputs(func_outputs)
    return model
