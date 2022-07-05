from typing import List
import pandas as pd

from summer import CompartmentalModel

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.model_features.random_process import RandomProcessProc
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.core.utils.utils import multiply_function_or_constant
from autumn.model_features.computed_values import FunctionWrapper
from autumn.model_features.random_process import get_random_process
from .outputs import SmCovidOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
# from .stratifications.agegroup import get_agegroup_strat
# from .stratifications.immunity import (
#     get_immunity_strat,
#     adjust_susceptible_infection_without_strains,
#     adjust_susceptible_infection_with_strains,
#     adjust_reinfection_without_strains,
#     adjust_reinfection_with_strains,
#     apply_reported_vacc_coverage,
#     apply_reported_vacc_coverage_with_booster,
# )

from autumn.models.sm_sir.stratifications.agegroup import convert_param_agegroups
from autumn.settings.constants import COVID_BASE_DATETIME

# Base date used to calculate mixing matrix times
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def build_model(
        params: dict,
        build_options: dict = None
) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    params = Parameters(**params)

    country = params.country
    pop = params.population
    iso3 = country.iso3
    region = pop.region
    age_groups = [str(age) for age in params.age_groups]
    # age_strat_params = params.age_stratification
    sojourns = params.sojourns
    
    time_params = params.time
    time_to_event_params = params.time_from_onset_to_event

    # Determine the infectious compartment(s)
    infectious_compartments = [Compartment.INFECTIOUS]

    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=infectious_compartments,
        timestep=time_params.step,
        ref_date=COVID_BASE_DATETIME,
    )

    """
    Check build options
    """

    # This will be automatically populated by calibration.py if we are running a calibration, but can be manually set
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    """
    Create the total population
    """

    # Get country population by age-group
    age_pops = pd.Series(
        inputs.get_population_by_agegroup(age_groups, iso3, region, pop.year),
        index=age_groups
    )

    # Assign the population to compartments
    susceptible_pop = age_pops.sum() - params.infectious_seed
    init_pop = {
        Compartment.INFECTIOUS: params.infectious_seed,
        Compartment.SUSCEPTIBLE: susceptible_pop,
    }

    # Assign to the model
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows
    """
    # From latent to active infection
    progression_rate = 1. / sojourns.latent
    model.add_transition_flow(
        name=FlowName.PROGRESSION,
        fractional_rate=progression_rate,
        source=Compartment.LATENT,
        dest=Compartment.INFECTIOUS,
    )

    # Transmission
    infection_dest, infectious_entry_flow = Compartment.LATENT, FlowName.PROGRESSION

    if params.activate_random_process:

        # Store random process as a computed value to make it available as an output
        rp_function, contact_rate = get_random_process(
            params.random_process,
            params.contact_rate
        )
        model.add_computed_value_process(
            "transformed_random_process",
            RandomProcessProc(rp_function)
        )

    else:
        contact_rate = params.contact_rate

    # Add the process of infecting the susceptibles
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Add recovery flow 
    recovery_rate = 1. / sojourns.active
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=Compartment.INFECTIOUS,
        dest=Compartment.RECOVERED,
    )

    """
    Apply age stratification
    """

    # # Preprocess age-specific parameters to match model age bands if requested in this way
    # if type(suscept_req) == dict:
    #     suscept_adjs = convert_param_agegroups(iso3, region, suscept_req, age_groups)
    # else:
    #     suscept_adjs = suscept_req  # In which case it should be None or a float, confirmed in parameter validation

    # if type(sympt_req) == dict:
    #     sympt_props = convert_param_agegroups(iso3, region, sympt_req, age_groups)
    #     sympt_props.index = sympt_props.index.map(str)  # Change int indices to string to match model format
    # else:
    #     sympt_props = sympt_req  # In which case it should be None or a float

    # # Get the age-specific mixing matrices
    # mixing_matrices = build_synthetic_matrices(
    #     iso3,
    #     params.ref_mixing_iso3,
    #     [int(age) for age in age_groups],
    #     True,  # Always age-adjust, could change this to being a parameter
    #     region
    # )

    # # Get the actual age stratification now
    # age_strat = get_agegroup_strat(
    #     params,
    #     age_groups,
    #     age_pops,
    #     mixing_matrices,
    #     compartment_types,
    #     params.is_dynamic_mixing_matrix,
    #     suscept_adjs,
    # )
    # model.stratify_with(age_strat)

    # """
    # Immunity stratification
    # """

    # # Get the immunity stratification
    # immunity_params = params.immunity_stratification
    # immunity_strat = get_immunity_strat(
    #     BASE_COMPARTMENTS,
    #     immunity_params,
    # )

    # # Adjust infection of susceptibles for immunity status
    # reinfection_flows = [FlowName.EARLY_REINFECTION] if voc_params else []
    # if Compartment.WANED in compartment_types:
    #     reinfection_flows.append(FlowName.LATE_REINFECTION)

    # immunity_low_risk_reduction = immunity_params.infection_risk_reduction.low
    # immunity_high_risk_reduction = immunity_params.infection_risk_reduction.high

    # if voc_params:
    #     # The code should run fine if VoC parameters have been submitted but the strain stratification hasn't been
    #     # implemented yet - but at this stage we assume we don't want it to
    #     msg = "Strain stratification not present in model"
    #     assert "strain" in [strat.name for strat in model._stratifications], msg
    #     adjust_susceptible_infection_with_strains(
    #         immunity_low_risk_reduction,
    #         immunity_high_risk_reduction,
    #         immunity_strat,
    #         voc_params,
    #     )
    #     adjust_reinfection_with_strains(
    #         immunity_low_risk_reduction,
    #         immunity_high_risk_reduction,
    #         immunity_strat,
    #         reinfection_flows,
    #         voc_params,
    #     )
    # else:
    #     adjust_susceptible_infection_without_strains(
    #         immunity_low_risk_reduction,
    #         immunity_high_risk_reduction,
    #         immunity_strat,
    #     )
    #     adjust_reinfection_without_strains(
    #         immunity_low_risk_reduction,
    #         immunity_high_risk_reduction,
    #         immunity_strat,
    #         reinfection_flows,
    #     )

    # # Apply the immunity stratification
    # model.stratify_with(immunity_strat)

    # # Implement the dynamic immunity process
    # vacc_coverage_available = ["BGD", "PHL", "BTN", "VNM"]
    # vacc_region_available = ["Metro Manila", "Hanoi", "Ho Chi Minh City", None]
    # is_dynamic_immunity = iso3 in vacc_coverage_available and region in vacc_region_available

    # if is_dynamic_immunity:
    #     thinning = 20 if iso3 == "BGD" else None

    #     if iso3 == "PHL" or iso3 == "VNM":
    #         apply_reported_vacc_coverage_with_booster(
    #             compartment_types,
    #             model,
    #             age_groups,
    #             iso3,
    #             region,
    #             thinning=thinning,
    #             model_start_time=params.time.start,
    #             start_immune_prop=immunity_params.prop_immune,
    #             start_prop_high_among_immune=immunity_params.prop_high_among_immune,
    #             booster_effect_duration=params.booster_effect_duration,
    #             future_monthly_booster_rate=params.future_monthly_booster_rate,
    #             future_booster_age_allocation=params.future_booster_age_allocation,
    #             age_pops=age_pops,
    #             model_end_time=params.time.end
    #         )
    #     else:
    #         apply_reported_vacc_coverage(
    #             compartment_types,
    #             model,
    #             iso3,
    #             thinning=thinning,
    #             model_start_time=params.time.start,
    #             start_immune_prop=immunity_params.prop_immune,
    #             additional_immunity_points=params.additional_immunity,
    #         )

    """
    Get the applicable outputs
    """

    model_times = model.times

    outputs_builder = SmCovidOutputsBuilder(model, BASE_COMPARTMENTS)
    
    outputs_builder.request_incidence(infectious_entry_flow)

    return model
