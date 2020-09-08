from copy import deepcopy

from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn import inputs
from autumn.curve import scale_up_function

from . import preprocess, outputs
from .validate import validate_params
from .stratification import stratify_by_organ, stratify_by_age


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a simple SIR model
    """
    validate_params(params)

    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    ]
    infectious_comps = [
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
    ]

    # prepare infectiousness adjustment for individuals on treatment
    treatment_infectiousness_adjustment = [
        {
            "comp_name": Compartment.ON_TREATMENT,
            "comp_strata": {},
            "value": params["on_treatment_infect_multiplier"],
        }
    ]

    # Define inter-compartmental flows.
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)

    # Set unstratified detection flow parameter
    params['detection_rate'] = params['passive_screening_rate'] * params['passive_screening_sensitivity']['unstratified']

    # Set unstratified treatment-outcome-related parameters
    mu = 1/60.  # FIXME: this should be a time-variant age-dependant quantity
    TSR = params['treatment_success_rate']
    params['treatment_recovery_rate'] = 1 / params['treatment_duration']
    params['treatment_death_rate'] = params['treatment_recovery_rate'] * (1. - TSR) / TSR *\
                                     params['prop_death_among_negative_tx_outcome'] /\
                                     (1. + params['prop_death_among_negative_tx_outcome']) -\
                                     mu
    params['relapse_rate'] = (params['treatment_death_rate'] + mu) / params['prop_death_among_negative_tx_outcome']

    # Define model times.
    integration_times = get_model_times_from_inputs(
        round(params["start_time"]), params["end_time"], params["time_step"]
    )

    # Define initial conditions - 1 infectious person.
    init_conditions = {Compartment.INFECTIOUS: 1}

    # load latency parameters
    if params["override_latency_rates"]:
        params = preprocess.latency.get_unstratified_parameter_values(params)

    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
            params["contact_rate"] * params["rr_infection_" + state]
        )

    # assign unstratified parameter values to infection death and self-recovery processes
    for param_name in ["infect_death_rate", "self_recovery_rate"]:
        params[param_name] = params[param_name + "_dict"]["unstratified"]

    # Create the model.
    tb_model = StratifiedModel(
        times=integration_times,
        compartment_names=compartments,
        initial_conditions=init_conditions,
        parameters=params,
        requested_flows=flows,
        infectious_compartments=infectious_comps,
        birth_approach=BirthApproach.ADD_CRUDE,
        entry_compartment=Compartment.SUSCEPTIBLE,
        starting_population=100000000,
    )

    # Add crude birth rate from UN estimates (using Federated States of Micronesia as a proxy as no data for RMI)
    birth_rates, years = inputs.get_crude_birth_rate(params['iso3'])
    tb_model.time_variants["crude_birth_rate"] = scale_up_function(
        years, birth_rates, smoothness=0.2, method=5
    )

    # Apply infectiousness adjustment for individuals on treatment
    tb_model.individual_infectiousness_adjustments = treatment_infectiousness_adjustment

    if "organ" in params["stratify_by"]:
        stratify_by_organ(tb_model, params)

    if "age" in params["stratify_by"]:
        stratify_by_age(tb_model, params, compartments)


    # Register derived output functions
    # These functions calculate 'derived' outputs of interest, based on the
    # model's compartment values or flows. These are calculated after the model is run.
    func_outputs = {
        "prevalence_infectious": outputs.calculate_prevalence_infectious,
        "prevalence_susceptible": outputs.calculate_prevalence_susceptible,
    }
    tb_model.add_function_derived_outputs(func_outputs)

    return tb_model
