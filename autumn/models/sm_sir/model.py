from datetime import date
from math import exp

from summer import CompartmentalModel

from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.tools.random_process import RandomProcess
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from .outputs import SmSirOutputsBuilder
from .parameters import Parameters
from .computed_values.random_process_compute import RandomProcessProc
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
from .stratifications.agegroup import get_agegroup_strat
from .stratifications.immunity import get_immunity_strat
from .stratifications.strains import get_strain_strat
from .stratifications.clinical import get_clinical_strat
from .strat_processing.strains import make_voc_seed_func
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

    # Preprocess age-specific parameters to match model age bands
    age_strat_params = params.age_stratification
    age_groups = params.age_groups
    sympt_props = age_strat_params.prop_symptomatic
    sympt_props = convert_param_agegroups(sympt_props, country.iso3, pop.region, age_groups) if sympt_props else None

    compartments = BASE_COMPARTMENTS
    if params.sojourns.exposed:
        compartments.append(Compartment.EXPOSED)
        if params.sojourns.exposed.proportion_early:
            compartments.append(Compartment.EXPOSED_LATE)
    if params.sojourns.active.proportion_early:
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
    total_pops = inputs.get_population_by_agegroup(age_groups, country.iso3, pop.region, pop.year)

    # Assign the remainder starting population to the S compartment
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows.
    """

    # Exposed compartment(s) transitions
    infection_dest = Compartment.INFECTIOUS
    infectious_entry_flow = FlowName.INFECTION
    if params.sojourns.exposed:
        exposed_sojourn = params.sojourns.exposed.total_time
        exposed_early_prop = params.sojourns.exposed.proportion_early

        if exposed_early_prop:
            early_sojourn = exposed_sojourn * exposed_early_prop
            model.add_transition_flow(
                name=FlowName.WITHIN_EXPOSED,
                fractional_rate=1. / early_sojourn,
                source=Compartment.EXPOSED,
                dest=Compartment.EXPOSED_LATE,
            )

            prop_exposed_late = 1. - exposed_early_prop
            progress_origin = Compartment.EXPOSED_LATE
            progress_rate = 1. / exposed_sojourn / prop_exposed_late

        else:
            progress_origin = Compartment.EXPOSED
            progress_rate = 1. / exposed_sojourn

        infection_dest = Compartment.EXPOSED
        model.add_transition_flow(
            name=FlowName.PROGRESSION,
            fractional_rate=progress_rate,
            source=progress_origin,
            dest=Compartment.INFECTIOUS,
        )
        infectious_entry_flow = FlowName.PROGRESSION

    # Transmission
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

    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Active compartment(s) transitions
    active_sojourn = params.sojourns.active.total_time
    active_early_prop = params.sojourns.active.proportion_early

    if active_early_prop:
        model.add_transition_flow(
            name=FlowName.WITHIN_INFECTIOUS,
            fractional_rate=1. / active_sojourn / active_early_prop,
            source=Compartment.INFECTIOUS,
            dest=Compartment.INFECTIOUS_LATE,
        )

        prop_active_late = 1. - active_early_prop
        recovery_origin = Compartment.INFECTIOUS_LATE
        recovery_rate = 1. / active_sojourn / prop_active_late
    else:
        recovery_origin = Compartment.INFECTIOUS
        recovery_rate = 1. / active_sojourn

    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=recovery_origin,
        dest=Compartment.RECOVERED,
    )

    """
    Apply age stratification
    """

    mixing_matrices = build_synthetic_matrices(country.iso3, params.ref_mixing_iso3, age_groups, True, pop.region)
    age_strat = get_agegroup_strat(params, total_pops, mixing_matrices, compartments, is_dynamic_matrix=False)
    model.stratify_with(age_strat)

    """
    Apply clinical stratification - must come after age stratification if asymptomatic props being used
    """

    clinical_strat = get_clinical_strat(sympt_props, compartments, age_groups, infectious_entry_flow)
    model.stratify_with(clinical_strat)

    """
    Apply strains stratification
    """

    if params.voc_emergence:
        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params, compartments)
        model.stratify_with(strain_strat)

        # Use importation flows to seed VoC cases
        for voc_name, voc_values in voc_params.items():
            voc_seed_func = make_voc_seed_func(voc_values.entry_rate, voc_values.start_time, voc_values.seed_duration)
            model.add_importation_flow(
                f"seed_voc_{voc_name}",
                voc_seed_func,
                dest=Compartment.INFECTIOUS,  # Different from Covid model, because this compartment always present here
                dest_strata={"strain": voc_name},
                split_imports=True
            )

    """
    Apply immunity stratification
    """
    immunity_strat = get_immunity_strat(params, compartments)
    model.stratify_with(immunity_strat)

    """
    Set up derived output functions
    """
    
    outputs_builder = SmSirOutputsBuilder(model, compartments)
    outputs_builder.request_incidence(sympt_props, age_groups)
    outputs_builder.request_notifications(
        params.prop_symptomatic_infections_notified,
        params.time_from_onset_to_event.notification,
        model.times,
        age_groups
    )
    outputs_builder.request_hospitalisations(
        params.age_stratification.prop_hospital,
        params.immunity_stratification.hospital_risk_reduction,
        params.time_from_onset_to_event.hospitalisation,
        model.times,
        age_groups
    )

    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    return model


def set_up_random_process(start_time, end_time):
    return RandomProcess(order=2, period=30, start_time=start_time, end_time=end_time)
