from datetime import date
from math import exp
from typing import List

from summer import CompartmentalModel

from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.tools.random_process import RandomProcess
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from .outputs import SmSirOutputsBuilder
from .parameters import Parameters, Sojourns
from .computed_values.random_process_compute import RandomProcessProc
from .constants import BASE_COMPARTMENTS, Compartment, FlowName
from .stratifications.agegroup import get_agegroup_strat
from .stratifications.immunity import get_immunity_strat
from .stratifications.strains import get_strain_strat
from .stratifications.clinical import get_clinical_strat
from .strat_processing.strains import seed_vocs
from .preprocess.age_specific_params import convert_param_agegroups


# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def get_compartments(sojourns: Sojourns) -> List[str]:
    """
    Find the model compartments that are applicable to the parameter requests, based on the sojourn times structure.
        - Start with just the base SIR structure
        - Add in the latent compartment if there is a latent period request
        - Add in a second serial latent compartment if there is a proportion of the latent period early
        - Add in a second serial active compartment if there is a proportion of the active period early

    Args:
        sojourns: User requested sojourn times

    Returns:
        The names of the model compartments to be implemented.

    """

    # Make a copy, we really don't want to append to something that's meant to be a constant...
    compartments = BASE_COMPARTMENTS.copy()
    
    if sojourns.latent:
        compartments.append(Compartment.LATENT)
        if sojourns.latent.proportion_early:
            compartments.append(Compartment.LATENT_LATE)
    if sojourns.active.proportion_early:
        compartments.append(Compartment.INFECTIOUS_LATE)
    if sojourns.recovered:
        compartments.append(Compartment.WANED)

    return compartments


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

    hosp_props = age_strat_params.prop_hospital
    hosp_props = convert_param_agegroups(hosp_props, country.iso3, pop.region, age_groups) if hosp_props else None

    # Determine the compartments
    base_compartments = get_compartments(params.sojourns)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=base_compartments,
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

    # Latent compartment(s) transitions
    if params.sojourns.latent:
        latent_sojourn = params.sojourns.latent.total_time
        latent_early_prop = params.sojourns.latent.proportion_early

        if latent_early_prop:
            early_sojourn = latent_sojourn * latent_early_prop
            model.add_transition_flow(
                name=FlowName.WITHIN_LATENT,
                fractional_rate=1. / early_sojourn,
                source=Compartment.LATENT,
                dest=Compartment.LATENT_LATE,
            )

            prop_latent_late = 1. - latent_early_prop
            progress_origin = Compartment.LATENT_LATE
            progress_rate = 1. / latent_sojourn / prop_latent_late

        else:
            progress_origin = Compartment.LATENT
            progress_rate = 1. / latent_sojourn

        model.add_transition_flow(
            name=FlowName.PROGRESSION,
            fractional_rate=progress_rate,
            source=progress_origin,
            dest=Compartment.INFECTIOUS,
        )

        infection_dest = Compartment.LATENT
        infectious_entry_flow = FlowName.PROGRESSION
    else:
        infection_dest = Compartment.INFECTIOUS
        infectious_entry_flow = FlowName.INFECTION

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

    # Infection flows
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )
    model.add_infection_frequency_flow(
        name=FlowName.EARLY_REINFECTION,
        contact_rate=contact_rate,
        source=Compartment.RECOVERED,
        dest=infection_dest,
    )
    if "waned" in base_compartments:
        model.add_infection_frequency_flow(
            name=FlowName.LATE_REINFECTION,
            contact_rate=contact_rate,
            source=Compartment.WANED,
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

    # Recovery and waning
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=recovery_origin,
        dest=Compartment.RECOVERED,
    )
    if "waned" in base_compartments:
        model.add_transition_flow(
            name=FlowName.WANING,
            fractional_rate=1. / params.sojourns.recovered.total_time,
            source=Compartment.RECOVERED,
            dest=Compartment.WANED,
        )

    """
    Apply age stratification
    """

    mixing_matrices = build_synthetic_matrices(
        country.iso3,
        params.ref_mixing_iso3,
        age_groups,
        True,  # Always age-adjust, could change this to being a parameter
        pop.region
    )
    age_strat = get_agegroup_strat(
        params,
        total_pops,
        mixing_matrices,
        base_compartments,
        params.is_dynamic_mixing_matrix
    )
    model.stratify_with(age_strat)

    """
    Apply clinical stratification - must come after age stratification if asymptomatic props being used
    """

    detect_prop = params.detect_prop
    is_undetected = callable(detect_prop) or detect_prop < 1.0
    if sympt_props:
        msg = "Attempted to apply differential symptomatic proportions by age, but model not age stratified"
        model_stratifications = [model._stratifications[i].name for i in range(len(model._stratifications))]
        assert "agegroup" in model_stratifications, msg

    if is_undetected or sympt_props:
        clinical_strat = get_clinical_strat(
            model, base_compartments, params, age_groups, infectious_entry_flow, detect_prop, is_undetected, sympt_props
        )
        model.stratify_with(clinical_strat)
        clinical_strata = clinical_strat.strata
    else:
        clinical_strata = None

    """
    Apply strains stratification
    """

    if params.voc_emergence:

        voc_params = params.voc_emergence

        # Build and apply stratification
        strain_strat = get_strain_strat(voc_params, base_compartments)
        model.stratify_with(strain_strat)

        # Seed the VoCs from the point in time
        seed_vocs(model, voc_params, Compartment.INFECTIOUS)

        strain_strata = strain_strat.strata

    else:
        strain_strata = [""]

    """
    Apply immunity stratification
    """

    immunity_strat = get_immunity_strat(
        base_compartments,
        params.immunity_stratification,
        strain_strata,
        params.voc_emergence,
    )
    model.stratify_with(immunity_strat)

    """
    Set up derived output functions
    """

    outputs_builder = SmSirOutputsBuilder(model, base_compartments)

    # Track CDR function if case detection is implemented
    if is_undetected:
        outputs_builder.request_cdr()

    outputs_builder.request_incidence(base_compartments, age_groups, clinical_strata, strain_strata)
    outputs_builder.request_notifications(
        params.time_from_onset_to_event.notification,
        model.times
    )
    outputs_builder.request_hospitalisations(
        hosp_props,
        params.hospital_prop_multiplier,
        params.immunity_stratification.hospital_risk_reduction,
        params.time_from_onset_to_event.hospitalisation,
        params.hospital_stay.hospital_all,
        model.times,
        age_groups
    )
    outputs_builder.request_icu_outputs(
        params.prop_icu_among_hospitalised,
        params.time_from_onset_to_event.icu_admission,
        params.hospital_stay.icu,
        model.times
    )

    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    return model


def set_up_random_process(start_time, end_time):
    return RandomProcess(order=2, period=30, start_time=start_time, end_time=end_time)
