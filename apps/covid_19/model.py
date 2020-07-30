
from summer.model import StratifiedModel
from summer.model.utils.string import find_all_strata

from autumn.tool_kit.utils import repeat_list_elements

from autumn.curve import tanh_based_scaleup

from autumn.tool_kit.utils import normalise_sequence
from autumn.constants import BirthApproach
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn import inputs
from autumn.environment.seasonality import get_seasonal_forcing
from apps.covid_19.constants import Compartment

from . import outputs, preprocess
from .stratification import stratify_by_clinical
from .validate import validate_params

import copy


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run the TB model for Covid-19
    """
    validate_params(params)

    detect_prop_params = params["time_variant_detection"]
    import_representative_age = params["import_representative_age"]
    import_times = params["data"]["times_imported_cases"]
    import_cases = params["data"]["n_imported_cases"]
    agegroup_max = params["agegroup_breaks"][0]
    agegroup_step = params["agegroup_breaks"][1]
    agegroup_strata = list(range(0, agegroup_max, agegroup_step))
    start_time = params["start_time"]
    end_time = params["end_time"]
    time_step = params["time_step"]

    # Look up the country population size by age-group, using UN data
    country_iso3 = params["iso3"]
    region = params["region"]
    total_pops = inputs.get_population_by_agegroup(agegroup_strata, country_iso3, region, year=2020)
    life_expectancy = inputs.get_life_expectancy_by_agegroup(agegroup_strata, country_iso3)[0]
    life_expectancy_latest = [life_expectancy[agegroup][-1] for agegroup in life_expectancy]

    # Define compartments
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_EXPOSED,
        Compartment.LATE_EXPOSED,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        Compartment.RECOVERED,
    ]

    # Indicate whether the compartments representing active disease are infectious
    is_infectious = {
        Compartment.EARLY_EXPOSED: False,
        Compartment.LATE_EXPOSED: True,
        Compartment.EARLY_INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }

    # Calculate compartment periods
    # FIXME: Needs tests.
    base_compartment_periods = params["compartment_periods"]
    compartment_periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_compartment_periods, compartment_periods_calc
    )

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    compartment_exit_flow_rates = {}
    for compartment in compartment_periods:
        param_key = f"within_{compartment}"
        compartment_exit_flow_rates[param_key] = 1.0 / compartment_periods[compartment]

    # Distribute infectious seed across infectious compartments
    infectious_seed = params["infectious_seed"]
    total_disease_time = sum([compartment_periods[c] for c in is_infectious])
    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_disease_time for c in is_infectious
    }

    # Force the remainder starting population to go to S compartment (Required as entry_compartment is late_infectious)
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

    # Set integration times
    integration_times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    # Add inter-compartmental transition flows
    flows = copy.deepcopy(preprocess.flows.DEFAULT_FLOWS)

    # Choose a birth approach
    implement_importation = params["implement_importation"]
    birth_approach = BirthApproach.ADD_CRUDE if implement_importation else BirthApproach.NO_BIRTH

    # Build mixing matrix.
    static_mixing_matrix = preprocess.mixing_matrix.build_static(country_iso3)
    dynamic_mixing_matrix = None
    dynamic_location_mixing_params = params["mixing"]
    dynamic_age_mixing_params = params["mixing_age_adjust"]
    microdistancing = params["microdistancing"]
    smooth_google_data = params["smooth_google_data"]

    if dynamic_location_mixing_params or dynamic_age_mixing_params:
        npi_effectiveness_params = params["npi_effectiveness"]
        google_mobility_locations = params["google_mobility_locations"]
        is_periodic_intervention = params.get("is_periodic_intervention")
        periodic_int_params = params.get("periodic_intervention")
        dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
            country_iso3,
            region,
            dynamic_location_mixing_params,
            dynamic_age_mixing_params,
            npi_effectiveness_params,
            google_mobility_locations,
            is_periodic_intervention,
            periodic_int_params,
            end_time,
            microdistancing,
            smooth_google_data,
        )

    # FIXME: Remove params from model_parameters
    model_parameters = {**params, **compartment_exit_flow_rates}
    model_parameters["to_infectious"] = model_parameters["within_" + Compartment.LATE_EXPOSED]

    model_parameters['immunity_loss_rate'] = 1. / params['immunity_duration']

    # implement waning immunity
    if not params['full_immunity']:

        flows.append(
            {
                "type": 'standard_flows',
                "origin": Compartment.RECOVERED,
                "to": Compartment.SUSCEPTIBLE,
                "parameter": "immunity_loss_rate",
            }
        )

    # Instantiate SUMMER model
    model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach=birth_approach,
        entry_compartment=Compartment.LATE_INFECTIOUS,  # to model imported cases
        starting_population=sum(total_pops),
        infectious_compartment=[i_comp for i_comp in is_infectious if is_infectious[i_comp]],
    )
    if dynamic_mixing_matrix:
        model.find_dynamic_mixing_matrix = dynamic_mixing_matrix
        model.dynamic_mixing_matrix = True

    # Implement seasonal forcing if requested, making contact rate a time-variant rather than constant
    if model_parameters["seasonal_force"]:
        seasonal_forcing_function = \
            get_seasonal_forcing(
                365., 173., model_parameters["seasonal_force"], model_parameters["contact_rate"]
            )
        model.time_variants["contact_rate"] = \
            seasonal_forcing_function
        model.adaptation_functions["contact_rate"] = \
            seasonal_forcing_function
        model.parameters["contact_rate"] = \
            "contact_rate"

    # Detected and symptomatic proportions primarily needed for the clinical stratification
    # - except for the following function

    # Create function describing the proportion of cases detected over time
    def detected_proportion(t):

        # Function representing the proportion of symptomatic people detected over time
        base_prop_detect = \
            tanh_based_scaleup(
                detect_prop_params["maximum_gradient"],
                detect_prop_params["max_change_time"],
                detect_prop_params["start_value"],
                detect_prop_params["end_value"]
            )

        # Return value modified for any future intervention that narrows the case detection gap
        int_detect_gap_reduction = model_parameters['int_detection_gap_reduction']
        return base_prop_detect(t) + (1. - base_prop_detect(t)) * int_detect_gap_reduction

    # Age dependent proportions of infected people who become symptomatic
    # This is defined 8x10 year bands, 0-70+, which we transform into 16x5 year bands 0-75+
    symptomatic_props = \
        repeat_list_elements(2, model_parameters["symptomatic_props"])

    def modelled_abs_detection_proportion_imported(t):
        return symptomatic_props[agegroup_strata.index(import_representative_age)] * \
               detected_proportion(t)

    # Set time-variant importation rate
    if implement_importation:
        import_rate_func = \
            preprocess.importation.get_importation_rate_func_as_birth_rates(
                import_times, import_cases, modelled_abs_detection_proportion_imported, total_pops,
            )
        model.parameters["crude_birth_rate"] = "crude_birth_rate"
        model.time_variants["crude_birth_rate"] = import_rate_func

    # Stratify model by age
    # Coerce age breakpoint numbers into strings - all strata are represented as strings
    agegroup_strings = [str(s) for s in agegroup_strata]
    # Create parameter adjustment request for age stratifications
    age_based_susceptibility = params["age_based_susceptibility"]
    adjust_requests = {
        # No change, but distinction is required for later stratification by clinical status
        "to_infectious": {s: 1 for s in agegroup_strings},
        "infect_death": {s: 1 for s in agegroup_strings},
        "within_late_active": {s: 1 for s in agegroup_strings},
        # Adjust susceptibility across age groups
        "contact_rate": age_based_susceptibility,
    }

    # Distribute starting population over agegroups
    requested_props = {
        agegroup: prop for agegroup, prop in zip(agegroup_strings, normalise_sequence(total_pops))
    }

    # We use "agegroup" instead of "age" for this model, to avoid triggering automatic demography features
    # (which work on the assumption that the time unit is years, so would be totally wrong)
    model.stratify(
        "agegroup",
        agegroup_strings,
        compartment_types_to_stratify=[],  # Apply to all compartments
        requested_proportions=requested_props,
        mixing_matrix=static_mixing_matrix,
        adjustment_requests=adjust_requests,
        entry_proportions=model_parameters["importation_props_by_age"],
    )

    model_parameters["all_stratifications"] = {"agegroup": agegroup_strings}

    # Allow pre-symptomatics to be less infectious
    model.individual_infectiousness_adjustments = \
        [
            [[Compartment.LATE_EXPOSED], model_parameters[Compartment.LATE_EXPOSED + "_infect_multiplier"]]
        ]

    # Stratify by clinical
    stratify_by_clinical(model, model_parameters, compartments, detected_proportion, symptomatic_props)

    # Define output connections to collate
    # Track compartment output connections.
    stratum_names = list(set([find_all_strata(x) for x in model.compartment_names]))
    incidence_connections = outputs.get_incidence_connections(stratum_names)
    progress_connections = outputs.get_progress_connections(stratum_names)
    model.output_connections = {
        **incidence_connections,
        **progress_connections,
    }

    # Add notifications to derived_outputs
    model.derived_output_functions["notifications"] = \
        outputs.get_calc_notifications_covid(implement_importation, modelled_abs_detection_proportion_imported)
    model.derived_output_functions["local_notifications"] = \
        outputs.get_calc_notifications_covid(False, modelled_abs_detection_proportion_imported)
    model.derived_output_functions["prevXlate_activeXclinical_icuXamong"] = \
        outputs.calculate_icu_prev
    model.derived_output_functions["new_hospital_admissions"] = \
        outputs.calculate_new_hospital_admissions_covid
    model.derived_output_functions["hospital_occupancy"] = \
        outputs.calculate_hospital_occupancy
    model.derived_output_functions["proportion_seropositive"] = \
        outputs.calculate_proportion_seropositive
    model.derived_output_functions["new_icu_admissions"] = \
        outputs.calculate_new_icu_admissions_covid
    model.derived_output_functions["icu_occupancy"] = \
        outputs.calculate_icu_occupancy
    model.death_output_categories = \
        list_all_strata_for_mortality(model.compartment_names)
    model.derived_output_functions["years_of_life_lost"] = \
        outputs.get_calculate_years_of_life_lost(life_expectancy_latest)

    return model
