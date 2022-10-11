from typing import List
import pandas as pd

from summer import CompartmentalModel, Multiply

from autumn.core import inputs
from autumn.core.project import Params, build_rel_path
from autumn.model_features.random_process import RandomProcessProc
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.core.utils.utils import multiply_function_or_constant
from autumn.model_features.computed_values import FunctionWrapper
from autumn.model_features.random_process import get_random_process
from .outputs import SmCovidOutputsBuilder
from .parameters import Parameters, Sojourns, CompartmentSojourn
from .constants import Compartment, REPLICABLE_COMPARTMENTS, FlowName
from .stratifications.immunity import (
    get_immunity_strat,
    adjust_susceptible_infection_without_strains,
    set_dynamic_vaccination_flows
)
from .stratifications.strains import get_strain_strat, seed_vocs_using_gisaid, apply_reinfection_flows_with_strains, adjust_reinfection_with_strains, adjust_susceptible_infection_with_strains

# Import modules from sm_sir model
from autumn.models.sm_sir.stratifications.agegroup import convert_param_agegroups, get_agegroup_strat

from autumn.settings.constants import COVID_BASE_DATETIME

# Base date used to calculate mixing matrix times
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def set_infectious_seed(model, infectious_seed_time, seed_duration, infectious_seed, dest_compartment):

    entry_rate = infectious_seed / seed_duration

    def infectious_seed_func(time: float, computed_values):
        return entry_rate if 0. < time - infectious_seed_time < seed_duration else 0.  

    model.add_importation_flow(
        FlowName.PRIMARY_INFECTIOUS_SEEDING,
        infectious_seed_func,
        dest=dest_compartment,
        split_imports=True
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
        table_name='school_closure', 
        conditions= {"country_id": params.country.iso3},
        columns=["date", "status", "weeks_partially_open", "weeks_fully_closed", "enrolment_(pre-primary_to_upper_secondary)"]
    )

    # Update the school mixing data if requested
    if params.mobility.apply_unesco_school_data:
        # Convert the categorical data into a numerical timeseries
        unesco_data.replace(
            {
                "Academic break": 0., 
                "Fully open": 1.,                 
                "Partially open": params.mobility.unesco_partial_opening_value,  # switched to 1. to model counterfactual no-closure scenario
                "Closed due to COVID-19": params.mobility.unesco_full_closure_value,  # switched to 1. to model counterfactual no-closure scenario
            }, 
            inplace=True
        )

        # Remove rows where status values are repeated (only keeps first and last timepoints for each plateau phase)
        unesco_data = unesco_data[unesco_data["status"].diff(periods=-1).diff() != 0]  

        # Override the baseline school mixing data with the UNESCO timeseries
        params.mobility.mixing["school"].append = False  # to override the baseline data rather than append 
        params.mobility.mixing["school"].times = (pd.to_datetime(unesco_data["date"])- COVID_BASE_DATETIME).dt.days.to_list()
        params.mobility.mixing["school"].values = unesco_data["status"].to_list()

    # read n_weeks_closed, n_weeks_partial
    n_weeks_closed, n_weeks_partial = unesco_data['weeks_fully_closed'].max(), unesco_data['weeks_partially_open'].max()
    n_students = unesco_data['enrolment_(pre-primary_to_upper_secondary)'].iloc[0]
    student_weeks_missed = n_students * (n_weeks_closed + n_weeks_partial * (1. - params.mobility.unesco_partial_opening_value))

    return student_weeks_missed

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
    age_strat_params = params.age_stratification
    sojourns = params.sojourns
    voc_params = params.voc_emergence
    time_params = params.time
    time_to_event_params = params.time_from_onset_to_event

    # Determine the lists of latent and infectious compartments based on replicate requests
    n_latent_comps = params.compartment_replicates['latent']
    latent_compartments = [f"{Compartment.LATENT}_{i}" for i in range(n_latent_comps)]

    n_active_comps = params.compartment_replicates['infectious']
    active_compartments = [f"{Compartment.INFECTIOUS}_{i}" for i in range(n_active_comps)]

    # Define the full list of compartments
    base_compartments = [Compartment.SUSCEPTIBLE] + latent_compartments + active_compartments + [Compartment.RECOVERED]

    # work out the list of infectious compartments
    n_infectious_latent_comps = params.latency_infectiousness.n_infectious_comps
    assert n_infectious_latent_comps <= n_latent_comps, "Number of infectious latent comps greater than number of latent comps."
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
        inputs.get_population_by_agegroup(age_groups, iso3, region, pop.year),
        index=age_groups
    )

    # Assign the population to compartments
    total_pop = age_pops.sum()
    infectious_seed = total_pop * 10 / 1.e6  # seed prevalence of 10 per million population
    susceptible_pop = total_pop - infectious_seed  # to leave room for the infectious seed
    init_pop = {
        Compartment.SUSCEPTIBLE: susceptible_pop,
    }

    # Assign to the model
    model.set_initial_population(init_pop)

    # Set infectious seed
    set_infectious_seed(model, params.infectious_seed_time, params.seed_duration, infectious_seed, infectious_compartments[0])

    """
    Add intercompartmental flows
    """
    # Transition within latent states 
    progression_rate = n_latent_comps * 1. / sojourns.latent
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

    # Transition within active states 
    recovery_rate = n_active_comps * 1. / sojourns.active
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
        sympt_props.index = sympt_props.index.map(str)  # Change int indices to string to match model format
    else:
        sympt_props = sympt_req  # In which case it should be None or a float

    # Get the age-specific mixing matrices
    mixing_matrices = build_synthetic_matrices(
        iso3,
        params.ref_mixing_iso3,
        [int(age) for age in age_groups],
        True,  # Always age-adjust, could change this to being a parameter
        region
    )

    # Apply UNESCO school closure data. This will update the school mixing params
    student_weeks_missed = process_unesco_data(params)

    # Get the actual age stratification now
    age_strat = get_agegroup_strat(
        params,
        age_groups,
        age_pops,
        mixing_matrices,
        base_compartments,
        params.is_dynamic_mixing_matrix,
        suscept_adjs,
    )

    # adjust the infectiousness of the infectious latent compartments using this stratification (summer design flaw, we should be able to do this with no stratification)
    for infectious_latent_comp in infectious_latent_comps:
        age_strat.add_infectiousness_adjustments(
            infectious_latent_comp, 
            {agegroup: Multiply(params.latency_infectiousness.rel_infectiousness) for agegroup in age_groups}
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
            {strain: Multiply(voc_params[strain].seed_prop) for strain in voc_params}
        )
        # Apply the stratification
        model.stratify_with(strain_strat)

        # Seed the VoCs from the requested point in time
        seed_vocs_using_gisaid(model, voc_params, infectious_compartments[0], country.country_name, infectious_seed)

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
    
    outputs_builder.request_incidence(age_groups, strain_strata, infectious_entry_flow, params.request_incidence_by_age)
    
    outputs_builder.request_hospitalisations(model_times, age_groups, strain_strata, iso3, region, age_strat_params.prop_symptomatic, age_strat_params.prop_hospital, params.vaccine_effects.ve_hospitalisation, time_to_event_params.hospitalisation, params.hospital_stay.hospital_all, voc_params)  
    outputs_builder.request_peak_hospital_occupancy()
    
    outputs_builder.request_infection_deaths(model_times, age_groups, strain_strata, iso3, region, age_strat_params.ifr, params.vaccine_effects.ve_death, time_to_event_params.death, voc_params)
    outputs_builder.request_recovered_proportion(base_compartments)
    outputs_builder.request_immunity_props(immunity_strat.strata, age_pops, params.request_immune_prop_by_age)

    outputs_builder.request_cumulative_outputs(params.requested_cumulative_outputs, params.cumulative_start_time)

    if params.activate_random_process:
        outputs_builder.request_random_process_outputs()

    # request extra output to store the number of students*weeks of school missed
    outputs_builder.request_student_weeks_missed_output(student_weeks_missed)

    return model
