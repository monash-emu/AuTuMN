from summer import CompartmentalModel
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.tools.random_process import RandomProcess
from math import exp
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices

from.outputs import SmSirOutputsBuilder
from .parameters import Parameters
from datetime import date
from .computed_values.random_process_compute import RandomProcessProc
from .constants import COMPARTMENTS, AGEGROUP_STRATA, Compartment, FlowName
from .stratifications.agegroup import get_agegroup_strat
from .stratifications.immunity import get_immunity_strat
from .preprocess.age_specific_params import convert_param_agegroups


# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """

    params = Parameters(**params)

    # Get country/region details
    country = params.country
    pop = params.population

    # preprocess age-specific parameters to match model age bands
    prop_symptomatic = convert_param_agegroups(params.age_stratification.prop_symptomatic, country.iso3, pop.region)
    prop_hospital = convert_param_agegroups(params.age_stratification.prop_hospital, country.iso3, pop.region)
    ifr = convert_param_agegroups(params.age_stratification.ifr, country.iso3, pop.region)

    compartments = COMPARTMENTS
    if params.sojourn.exposed:
        compartments.append(Compartment.EXPOSED)
    if params.sojourn.active.proportion_early:
        compartments.append(Compartment.INFECTIOUS_LATE)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=compartments,
        infectious_compartments=[Compartment.INFECTIOUS],
        timestep=params.time.step,
        ref_date=BASE_DATE
    )

    # Check build_options
    # This will be automatically populated by calibration.py if we are running a calibration,
    # but can be manually set if so desired
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    """
    Create the total population.
    """
    init_pop = {Compartment.INFECTIOUS: params.infectious_seed}

    # Get country population by age-group
    total_pops = inputs.get_population_by_agegroup(AGEGROUP_STRATA, country.iso3, pop.region, pop.year)

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows.
    """
    # transmission
    if params.activate_random_process:
        # build the random process, using default values and coefficients
        rp = set_up_random_process(params.time.start, params.time.end)

        # update random process details based on the model parameters
        rp.update_config_from_params(params.random_process)

        # Create function returning exp(W), where W is the random process
        rp_time_variant_func = rp.create_random_process_function(transform_func=lambda w: exp(w))

        # store random process as a computed value to make it available as an output
        model.add_computed_value_process(
            "transformed_random_process",
            RandomProcessProc(
                rp_time_variant_func
            )
        )        
        
        # Create the time-variant contact rate that uses our computed random process
        def contact_rate(t, computed_values):
            return params.contact_rate * computed_values["transformed_random_process"]

    else:
        contact_rate = params.contact_rate

    # Define destination of infection and include exposed compartment if requested
    infection_dest = Compartment.INFECTIOUS  # For use in the infection flow below
    if params.sojourn.exposed:
        infection_dest = Compartment.EXPOSED
        model.add_transition_flow(
            name=FlowName.PROGRESSION,
            fractional_rate=params.sojourn.exposed.total_time,
            source=Compartment.EXPOSED,
            dest=Compartment.INFECTIOUS,
        )

    # Include the infection process
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Recovery
    recovery_origin = Compartment.INFECTIOUS
    recovery_rate = 1. / params.sojourn.active.total_time
    if params.sojourn.active.proportion_early:
        recovery_origin = Compartment.INFECTIOUS_LATE
        prop_active_late = (1. - params.sojourn.active.proportion_early)
        recovery_rate = 1. / (params.sojourn.active.total_time * prop_active_late)
        model.add_transition_flow(
            name=FlowName.WITHIN_INFECTIOUS,
            fractional_rate=1. / (params.sojourn.active.total_time * params.sojourn.active.proportion_early),
            source=Compartment.INFECTIOUS,
            dest=Compartment.INFECTIOUS_LATE,
        )
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=recovery_origin,
        dest=Compartment.RECOVERED,
    )

    """
    Apply age stratification
    """
    mixing_matrices = build_synthetic_matrices(
        country.iso3, params.ref_mixing_iso3, AGEGROUP_STRATA, True, pop.region, requested_locations=["all_locations"]
    )

    age_strat = get_agegroup_strat(params, total_pops, mixing_matrices["all_locations"], compartments)
    model.stratify_with(age_strat)

    """
    Apply immunity stratification
    """
    immunity_strat = get_immunity_strat(params, compartments)
    model.stratify_with(immunity_strat)

    """
    Set up derived output functions
    """
    outputs_builder = SmSirOutputsBuilder(model, compartments)
    outputs_builder.request_incidence()
    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    """
    Calculate hospitalisations and deaths
    """
    # FIXME! Will need to use new type of derived outputs using delayed events triggered from other derived outputs.

    return model


def set_up_random_process(start_time, end_time):
    return RandomProcess(order=2, period=30, start_time=start_time, end_time=end_time)
