from copy import deepcopy

from autumn import inputs
from autumn.constants import Flow, BirthApproach
from autumn.curve import tanh_based_scaleup
from autumn.environment.seasonality import get_seasonal_forcing
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.utils import normalise_sequence, repeat_list_elements
from summer.model import StratifiedModel
from apps.covid_19.model.susceptibility_heterogeneity import get_gamma_data, check_modelled_susc_cv

from apps.covid_19.constants import Compartment
from apps.covid_19.model.importation import get_all_vic_notifications
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS, Region

from . import outputs, preprocess
from .stratification import stratify_by_clinical
from .preprocess.testing import find_cdr_function_from_test_data
from .validate import validate_params


def build_model(params: dict) -> StratifiedModel:
    """
    Build the compartmental model from the provided parameters.
    """
    validate_params(params)

    """
    Integration times
    """

    start_time, end_time, time_step = params["start_time"], params["end_time"], params["time_step"]
    times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    """
    Compartments
    """

    # Define infectious compartments
    infectious_comps = [
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_ACTIVE,
        Compartment.LATE_ACTIVE,
    ]

    # Extend to infected compartments
    disease_comps = [
        Compartment.EARLY_EXPOSED,
        *infectious_comps
    ]

    # Extent to all compartments
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
        *disease_comps
    ]

    """
    Basic intercompartmental flows
    """

    # Time periods calculated from periods (or "sojourn times")
    base_periods = params["compartment_periods"]
    periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_periods, periods_calc
    )

    # Inter-compartmental transition flows
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)
    flow_params = {
        "contact_rate": params["contact_rate"],
        "infect_death": 0,  # Placeholder to be overwritten in clinical stratification
    }

    # Add parameters for the during-disease progression flows
    for comp_name, comp_period in compartment_periods.items():
        flow_params[f"within_{comp_name}"] = 1.0 / comp_period

    # Waning immunity (if requested)
    if not params["full_immunity"]:
        flow_params["immunity_loss_rate"] = 1.0 / params["immunity_duration"]
        flows.append(
            {
                "type": Flow.STANDARD,
                "origin": Compartment.RECOVERED,
                "to": Compartment.SUSCEPTIBLE,
                "parameter": "immunity_loss_rate",
            }
        )

    # Just set the importation flow without specifying its value, which is done later (if requested)
    # Entry compartment is set to LATE_ACTIVE in the model creation process, because the default would be susceptible
    implement_importation = params["implement_importation"]
    if implement_importation:
        flows.append({"type": Flow.IMPORT, "parameter": "importation_rate"})

    """
    Population creation and distribution
    """

    # Set age groups
    agegroup_max, agegroup_step = params["agegroup_breaks"]
    agegroup_strata = list(range(0, agegroup_max, agegroup_step))

    # Get country population by age-group
    country_iso3 = params["iso3"]
    mobility_region = params["mobility_region"]
    pop_region = params["pop_region"]
    total_pops = inputs.get_population_by_agegroup(
        agegroup_strata, country_iso3, pop_region, year=params["pop_year"]
    )

    # Distribute infectious seed across infectious split sub-compartments
    infectious_seed = params["infectious_seed"]
    total_disease_time = sum([compartment_periods[c] for c in disease_comps])
    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_disease_time for c in disease_comps
    }

    # Assign the remainder starting population to the S compartment
    # (must be specified because entry_compartment is late_infectious)
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

    """
    Model instantiation
    """

    model = StratifiedModel(
        times,
        compartments,
        init_pop,
        flow_params,
        flows,
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.LATE_ACTIVE,
        starting_population=sum(total_pops),
        infectious_compartments=infectious_comps,
    )

    """
    Seasonal forcing
    """

    if params["seasonal_force"]:
        seasonal_func = get_seasonal_forcing(
            365.0, 173.0, params["seasonal_force"], params["contact_rate"]
        )
        model.time_variants["contact_rate"] = seasonal_func

    """
    Dynamic heterogeneous mixing by age
    """

    static_mixing_matrix = preprocess.mixing_matrix.build_static(country_iso3)
    dynamic_location_mixing_params = params["mixing"]
    dynamic_age_mixing_params = params["mixing_age_adjust"]
    microdistancing = params["microdistancing"]
    smooth_google_data = params["smooth_google_data"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    microdistancing_locations = params["microdistancing_locations"]
    dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
        country_iso3,
        mobility_region,
        dynamic_location_mixing_params,
        dynamic_age_mixing_params,
        npi_effectiveness_params,
        google_mobility_locations,
        microdistancing,
        smooth_google_data,
        microdistancing_locations,
    )
    model.set_dynamic_mixing_matrix(dynamic_mixing_matrix)

    """
    Age stratification
    """

    # Distribution of starting population (evenly over age groups)
    comp_split_props = {
        str(agegroup): prop
        for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }

    importation_props_by_age = \
        params["importation_props_by_age"] if \
            params["importation_props_by_age"] else \
            {str(agegroup_strata[i]): 1. / len(agegroup_strata) for i in range(len(agegroup_strata))}
    flow_adjustments = {
        "contact_rate": params["age_based_susceptibility"],
        "importation_rate": importation_props_by_age,
    }

    # Determine how many importations there are, including the undetected and asymptomatic importations
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    symptomatic_props = repeat_list_elements(2, params["symptomatic_props"])

    # We use "agegroup" instead of "age" for this model, to avoid triggering automatic demography features
    # (which also works on the assumption that the time unit is years, so would be totally wrong)
    model.stratify(
        "agegroup",
        agegroup_strata,
        compartments_to_stratify=compartments,
        comp_split_props=comp_split_props,
        flow_adjustments=flow_adjustments,
        mixing_matrix=static_mixing_matrix,
    )

    """
    Case detection
    """

    # More empiric approach based on per capita testing rates
    if params["testing_to_detection"]:
        assumed_tests_parameter = \
            params["testing_to_detection"]["assumed_tests_parameter"]
        assumed_cdr_parameter = \
            params["testing_to_detection"]["assumed_cdr_parameter"]  # Typically a calibration parameter

        # Use state denominator for testing rates for the Victorian health cluster models
        testing_region = "Victoria" if country_iso3 == "AUS" else pop_region
        testing_year = 2020 if country_iso3 == "AUS" else params["pop_year"]

        testing_pops = inputs.get_population_by_agegroup(
            agegroup_strata, country_iso3, testing_region, year=testing_year
        )

        detected_proportion = find_cdr_function_from_test_data(
            assumed_tests_parameter,
            assumed_cdr_parameter,
            country_iso3,
            testing_pops,
        )

    # Approach based on a hyperbolic tan function
    else:
        detect_prop_params = params["time_variant_detection"]

        def detected_proportion(t):
            return tanh_based_scaleup(
                detect_prop_params["maximum_gradient"],
                detect_prop_params["max_change_time"],
                detect_prop_params["start_value"],
                detect_prop_params["end_value"],
            )(t)

    """
    Importation 
    """

    import_symptomatic_prop = sum(
        [
            import_prop * sympt_prop
            for import_prop, sympt_prop in zip(
            importation_props_by_age.values(), symptomatic_props
        )
        ]
    )

    def modelled_abs_detection_proportion_imported(t):
        return import_symptomatic_prop * detected_proportion(t)

    if implement_importation:
        if (
                pop_region
                and pop_region.lower().replace("__", "_").replace("_", "-")
                in Region.VICTORIA_SUBREGIONS
        ):
            import_times, importation_data = get_all_vic_notifications(
                excluded_regions=(pop_region,)
            )
            movement_prop = params["movement_prop"]
            movement_to_region = sum(total_pops) / sum(testing_pops) * movement_prop
            import_cases = [i_cases * movement_to_region for i_cases in importation_data]

        else:
            import_times = params["data"]["times_imported_cases"]
            import_cases = params["data"]["n_imported_cases"]

        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times, import_cases, modelled_abs_detection_proportion_imported
        )
        model.time_variants["importation_rate"] = import_rate_func

    # Stratify the model by clinical status
    stratify_by_clinical(model, params, detected_proportion, symptomatic_props)

    """
    Susceptibility stratification
    """

    susceptibility_heterogeneity = params["susceptibility_heterogeneity"]

    if susceptibility_heterogeneity:
        tail_cut = susceptibility_heterogeneity["tail_cut"]
        bins = susceptibility_heterogeneity["bins"]
        coeff_var = susceptibility_heterogeneity["coeff_var"]

        # Interpret data requests
        _, _, susc_values, susc_pop_props, _ = get_gamma_data(tail_cut, bins, coeff_var)
        check_modelled_susc_cv(susc_values, susc_pop_props, coeff_var)

        # Define strata names
        susc_strata_names = [
            f"suscept_{i_susc}" for
            i_susc in range(bins)
        ]

        # Assign susceptibility values
        susc_adjustments = {
            susc_name: susc_value for
            susc_name, susc_value in zip(susc_strata_names, susc_values)
        }

        # Assign proportions of the population
        sus_pop_splits = {
            susc_name: susc_prop for
            susc_name, susc_prop in zip(susc_strata_names, susc_pop_props)
        }

        # Apply to all age groups individually (given current SUMMER API)
        susceptibility_adjustments = {
            f"contact_rateXagegroup_{str(i_agegroup)}": susc_adjustments for i_agegroup in agegroup_strata
        }

        # Stratify
        model.stratify(
            "suscept",
            list(susc_adjustments.keys()),
            [Compartment.SUSCEPTIBLE],
            flow_adjustments=susceptibility_adjustments,
            comp_split_props=sus_pop_splits,
        )

    """
    Set up and track derived output functions
    """

    # Set up derived outputs
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

    # Build life expectancy derived output function
    life_expectancy = inputs.get_life_expectancy_by_agegroup(agegroup_strata, country_iso3)[0]
    life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]
    life_lost_func = outputs.get_calculate_years_of_life_lost(life_expectancy_latest)

    # Build hospital occupancy func
    compartment_periods = params["compartment_periods"]
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    calculate_hospital_occupancy = outputs.get_calculate_hospital_occupancy(
        icu_early_period, hospital_early_period
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
    if mobility_region in OPTI_REGIONS:
        func_outputs["accum_years_of_life_lost"] = outputs.calculate_cum_years_of_life_lost

    model.add_function_derived_outputs(func_outputs)
    return model
