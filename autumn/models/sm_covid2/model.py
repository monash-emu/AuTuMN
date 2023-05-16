from typing import List
import pandas as pd

from summer2 import CompartmentalModel
from summer2.experimental.model_builder import ModelBuilder

from jax import numpy as jnp
from computegraph.types import Function

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.core.inputs.social_mixing.build_synthetic_matrices import get_matrices_from_conmat
from autumn.model_features.jax.random_process import get_random_process
from .outputs import SmCovidOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn
from .constants import Compartment, REPLICABLE_COMPARTMENTS, FlowName
from .stratifications.immunity import (
    get_immunity_strat,
    adjust_susceptible_infection_without_strains,
    set_dynamic_vaccination_flows,
)
from .stratifications.strains import (
    get_strain_strat,
    seed_vocs_using_gisaid,
    make_voc_seed_func,
    apply_reinfection_flows_with_strains,
    adjust_reinfection_with_strains,
    adjust_susceptible_infection_with_strains,
)

# Import modules from sm_sir model
from autumn.models.sm_jax.stratifications.agegroup import (
    convert_param_agegroups,
    get_agegroup_strat,
)

from autumn.settings.constants import COVID_BASE_DATETIME

from pathlib import Path


def get_base_params():
    base_params = Params(
        str(Path(__file__).parent.resolve() / "params.yml"),
        validator=lambda d: Parameters(**d),
        validate=False,
    )
    return base_params


def set_infectious_seed(
    model, infectious_seed_time, seed_duration, infectious_seed, dest_compartment
):

    entry_rate = infectious_seed / seed_duration

    infectious_seed_func = make_voc_seed_func(entry_rate, infectious_seed_time, seed_duration)

    model.add_importation_flow(
        FlowName.PRIMARY_INFECTIOUS_SEEDING,
        infectious_seed_func,
        dest=dest_compartment,
        split_imports=True,
    )


def process_unesco_data(params: Parameters):
    """_summary_
    Load school closure data from UNESCO and convert it into a timeseries object overriding the baseline school mixing data.

    Args:
        params: The model parameters

    """
    # Get the UNSECO school closures data
    input_db = inputs.database.get_input_db()
    unesco_data = input_db.query(
        table_name="school_closure",
        conditions={"country_id": params.country.iso3},
        columns=[
            "date",
            "status",
            "weeks_partially_open",
            "weeks_fully_closed",
            "enrolment_(pre-primary_to_upper_secondary)",
        ],
    )

    # remove rows with identical closure status to make dataframe lighter
    def map_func(key):
        return ['Fully open', 'Partially open', 'Closed due to COVID-19', 'Academic break'].index(key)

    unesco_data['status_coded'] = unesco_data['status'].transform(map_func)
    unesco_data = unesco_data[unesco_data["status_coded"].diff(periods=-1).diff() != 0]
    
    # Update the school mixing data if requested
    if params.mobility.apply_unesco_school_data:      
        # Convert the categorical data to a single timeseries representing the school attendance proportion

        # Build a function that we can embed in the model graph

        def build_unesco_mobility_array(partial_opening_value, full_closure_value):
            
            # Build this map inside the function so that we capture the arguments
            status_values = {
                "Academic break": 0.,
                "Fully open": 1.,
                "Partially open": partial_opening_value,
                "Closed due to COVID-19": full_closure_value
            }
            attendance_prop = jnp.zeros(len(unesco_data))
            
            for status, value in status_values.items():
                # Infill with value where we match this status
                # Note that it's ok to use this pandas series inside this JIT function, since it is not a dynamic argument
                # to the function, and we convert to a numpy type before being consumed by jax
                attendance_prop = jnp.where(unesco_data['status'].to_numpy() == status, value, attendance_prop)
            
            return attendance_prop
        
        # Wrap this as a computegraph Function so that it consumes the params
        # Note these can be full Parameter values (ie declared as pclass), or just constants
        attendance_prop_f = Function(build_unesco_mobility_array, [params.mobility.unesco_partial_opening_value,
                                               params.mobility.unesco_full_closure_value])
        
        # Override the baseline school mixing data with the UNESCO timeseries
        school_mobility_index = (
            pd.to_datetime(unesco_data["date"]) - COVID_BASE_DATETIME
        ).dt.days.to_numpy()

        school_mobility_data = attendance_prop_f

        additional_mobility = {
            "school": (
                school_mobility_index,
                school_mobility_data
            )
        }
    else:
        additional_mobility = {}

    # read n_weeks_closed, n_weeks_partial
    n_weeks_closed, n_weeks_partial = (
        unesco_data["weeks_fully_closed"].max(),
        unesco_data["weeks_partially_open"].max(),
    )
    n_students = unesco_data["enrolment_(pre-primary_to_upper_secondary)"].iloc[0]
    student_weeks_missed = n_students * (
        n_weeks_closed + n_weeks_partial * (1.0 - params.mobility.unesco_partial_opening_value)
    )

    return student_weeks_missed, additional_mobility


def scale_school_contacts(raw_matrices, school_multiplier):
    """
    Adjust the contact rates in schools according to the "school_multiplier" parameter.
    The "all_locations" matrix is also adjusted automatically to remain equal to the sum of the four 
    location-specific matrices.
    """
    # First, copy matrix values for "home", "work" and "other_locations" settings
    adjusted_matrices = {location: jnp.array(matrix) for location, matrix in raw_matrices.items() if location not in ['school', 'all_locations']}    
    
    # Then scale the school matrix according to the multiplier parameter
    adjusted_matrices['school'] = raw_matrices['school'] * school_multiplier
    adjusted_matrices['school'].node_name = "school_matrix"

    # We now need to compute the "all_locations" matrix, as the sum of all setting-specific matrices   
    from summer2.parameters import Data

    non_school = Data(adjusted_matrices['home'] + adjusted_matrices['work'] + adjusted_matrices['other_locations'])
    non_school.node_name = "non_school_matrix"
    adjusted_matrices['all_locations'] = non_school + adjusted_matrices['school']

    return adjusted_matrices


def build_model(params: dict, build_options: dict = None, ret_builder=False) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    builder = ModelBuilder(params, Parameters)
    params = builder.params

    country = params.country
    pop = params.population
    iso3 = country.iso3
    region = pop.region
    age_groups = [str(age) for age in params.age_groups]
    age_strat_params = params.age_stratification
    sojourns = params.sojourns
    voc_params = params.voc_emergence
    time_params = params.time
    time_to_event_params = params.time_from_onset_to_event

    # Determine the lists of latent and infectious compartments based on replicate requests
    n_latent_comps = params.compartment_replicates["latent"]
    latent_compartments = [f"{Compartment.LATENT}_{i}" for i in range(n_latent_comps)]

    n_active_comps = params.compartment_replicates["infectious"]
    active_compartments = [f"{Compartment.INFECTIOUS}_{i}" for i in range(n_active_comps)]

    # Define the full list of compartments
    base_compartments = (
        [Compartment.SUSCEPTIBLE]
        + latent_compartments
        + active_compartments
        + [Compartment.RECOVERED]
    )

    # work out the list of infectious compartments
    n_infectious_latent_comps = params.latency_infectiousness.n_infectious_comps
    assert (
        n_infectious_latent_comps <= n_latent_comps
    ), "Number of infectious latent comps greater than number of latent comps."
    infectious_latent_comps = latent_compartments[-n_infectious_latent_comps:]
    infectious_compartments = infectious_latent_comps + active_compartments

    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=base_compartments,
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
        inputs.get_population_by_agegroup(age_groups, iso3, region, pop.year), index=age_groups
    )

    # Assign the population to compartments
    total_pop = age_pops.sum()
    infectious_seed = total_pop * 10 / 1.0e6  # seed prevalence of 10 per million population
    susceptible_pop = total_pop - infectious_seed  # to leave room for the infectious seed
    init_pop = {
        Compartment.SUSCEPTIBLE: susceptible_pop,
    }

    # Assign to the model
    model.set_initial_population(init_pop)

    # Set infectious seed
    set_infectious_seed(
        model,
        params.infectious_seed_time,
        params.seed_duration,
        infectious_seed,
        infectious_compartments[0],
    )

    """
    Add intercompartmental flows
    """
    # Transition within latent states
    progression_rate = n_latent_comps * 1.0 / sojourns.latent
    latent_progression_flows = []
    for i_latent in range(n_latent_comps - 1):
        flown_name = f"within_latent_{i_latent}"
        model.add_transition_flow(
            name=flown_name,
            fractional_rate=progression_rate,
            source=latent_compartments[i_latent],
            dest=latent_compartments[i_latent + 1],
        )
        latent_progression_flows.append(flown_name)

    # Progression from latent to active
    model.add_transition_flow(
        name=FlowName.PROGRESSION,
        fractional_rate=progression_rate,
        source=latent_compartments[-1],
        dest=active_compartments[0],
    )
    latent_progression_flows.append(FlowName.PROGRESSION)

    # Transmission
    infection_dest, infectious_entry_flow = latent_compartments[0], FlowName.PROGRESSION
    contact_rate = params.contact_rate

    # Add the process of infecting the susceptibles
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=contact_rate,
        source=Compartment.SUSCEPTIBLE,
        dest=infection_dest,
    )

    # Transition within active states
    recovery_rate = n_active_comps * 1.0 / sojourns.active
    for i_active in range(n_active_comps - 1):
        model.add_transition_flow(
            name=f"within_infectious_{i_active}",
            fractional_rate=recovery_rate,
            source=active_compartments[i_active],
            dest=active_compartments[i_active + 1],
        )

    # Recovery transition
    model.add_transition_flow(
        name=FlowName.RECOVERY,
        fractional_rate=recovery_rate,
        source=infectious_compartments[-1],
        dest=Compartment.RECOVERED,
    )

    """
    Apply age stratification
    """
    suscept_req = age_strat_params.susceptibility
    sympt_req = age_strat_params.prop_symptomatic

    # Preprocess age-specific parameters to match model age bands if requested in this way
    if type(suscept_req) == dict:
        suscept_adjs = convert_param_agegroups(iso3, region, suscept_req, age_groups)
    else:
        suscept_adjs = suscept_req  # In which case it should be None or a float, confirmed in parameter validation

    if type(sympt_req) == dict:
        sympt_props = convert_param_agegroups(iso3, region, sympt_req, age_groups)
        sympt_props.index = sympt_props.index.map(
            str
        )  # Change int indices to string to match model format
    else:
        sympt_props = sympt_req  # In which case it should be None or a float

    # Get the age-specific mixing matrices using conmat R package
    raw_mixing_matrices = get_matrices_from_conmat(iso3, [int(age) for age in age_groups])
    # scale school contacts
    school_multiplier = params.school_multiplier    
    mixing_matrices = scale_school_contacts(raw_mixing_matrices, school_multiplier)

    # Apply UNESCO school closure data. This will update the school mixing params
    student_weeks_missed, additional_mobility = process_unesco_data(params)

    # Random process
    if params.activate_random_process:
        # Store random process as a computed value to make it available as an output
        rp_function = get_random_process(params.random_process)
        model.add_computed_value_func("transformed_random_process", rp_function)
    else:
        rp_function = None

    # Get the actual age stratification now
    age_strat = get_agegroup_strat(
        params,
        age_groups,
        age_pops,
        mixing_matrices,
        base_compartments,
        params.is_dynamic_mixing_matrix,
        suscept_adjs,
        additional_mobility,
        rp_function
    )

    #age_strat.set_mixing_matrix(mixing_matrices["all_locations"])

    # adjust the infectiousness of the infectious latent compartments using this stratification (summer design flaw, we should be able to do this with no stratification)
    for infectious_latent_comp in infectious_latent_comps:
        age_strat.add_infectiousness_adjustments(
            infectious_latent_comp,
            {agegroup: params.latency_infectiousness.rel_infectiousness for agegroup in age_groups},
        )

    model.stratify_with(age_strat)

    """
    Apply strains stratification
    """
    if voc_params:
        # Build the stratification using the same function as for the sm_sir model
        strain_strat = get_strain_strat(voc_params, base_compartments, latent_progression_flows)
        # Make sure the original infectious seed is split according to the strain-specific seed proportions
        strain_strat.set_flow_adjustments(
            FlowName.PRIMARY_INFECTIOUS_SEEDING,
            {strain: voc_params[strain].seed_prop for strain in voc_params},
        )
        # Apply the stratification
        model.stratify_with(strain_strat)

        # Seed the VoCs from the requested point in time
        seed_vocs_using_gisaid(
            model, voc_params, infectious_compartments[0], country.country_name, infectious_seed
        )

        # Keep track of the strain strata, which are needed for various purposes below
        strain_strata = strain_strat.strata

    # Need a placeholder for outputs and reinfection flows otherwise
    else:
        strain_strata = [""]

    """
    Apply the reinfection flows (knowing the strain stratification)
    """
    if voc_params:
        apply_reinfection_flows_with_strains(
            model,
            base_compartments,
            latent_compartments[0],
            age_groups,
            voc_params,
            strain_strata,
            contact_rate,
            suscept_adjs,
        )

    """
    Immunity stratification
    """

    # Get the immunity stratification
    vaccine_effects_params = params.vaccine_effects
    immunity_strat = get_immunity_strat(
        base_compartments,
    )

    # Adjust all transmission flows for immunity status and strain status (when relevant)
    ve_against_infection = vaccine_effects_params.ve_infection
    if voc_params:
        msg = "Strain stratification not present in model"
        assert "strain" in [strat.name for strat in model._stratifications], msg
        adjust_susceptible_infection_with_strains(
            ve_against_infection,
            immunity_strat,
            voc_params,
        )
        adjust_reinfection_with_strains(
            ve_against_infection,
            immunity_strat,
            voc_params,
        )
    else:
        adjust_susceptible_infection_without_strains(ve_against_infection, immunity_strat)

    # Apply the immunity stratification
    model.stratify_with(immunity_strat)

    # Apply dynamic vaccination flows
    set_dynamic_vaccination_flows(base_compartments, model, iso3, age_pops)

    """
    Get the applicable outputs
    """
    model_times = model.times

    outputs_builder = SmCovidOutputsBuilder(model, base_compartments)

    outputs_builder.request_incidence(
        age_groups, strain_strata, infectious_entry_flow, params.request_incidence_by_age
    )

    outputs_builder.request_hospitalisations(
        model_times,
        age_groups,
        strain_strata,
        iso3,
        region,
        age_strat_params.prop_symptomatic,
        age_strat_params.prop_hospital,
        params.vaccine_effects.ve_hospitalisation,
        time_to_event_params.hospitalisation,
        params.hospital_stay.hospital_all,
        voc_params,
    )

    outputs_builder.request_peak_hospital_occupancy()

    outputs_builder.request_infection_deaths(
        model_times,
        age_groups,
        strain_strata,
        iso3,
        region,
        age_strat_params.ifr,
        params.vaccine_effects.ve_death,
        time_to_event_params.death,
        voc_params,
    )

    outputs_builder.request_recovered_proportion(base_compartments)
    outputs_builder.request_age_matched_recovered_proportion(
        base_compartments, 
        age_groups, 
        params.serodata_age['min'], 
        params.serodata_age['max']
    )

    outputs_builder.request_immunity_props(
        immunity_strat.strata, age_pops, params.request_immune_prop_by_age
    )

    outputs_builder.request_cumulative_outputs(
        params.requested_cumulative_outputs, params.cumulative_start_time
    )

    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    # request extra output to store the number of students*weeks of school missed
    outputs_builder.request_student_weeks_missed_output(student_weeks_missed)

    builder.set_model(model)
    if ret_builder:
        return model, builder
    else:
        return model
