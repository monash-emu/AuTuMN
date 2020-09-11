from copy import deepcopy

from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.demography import set_model_time_variant_birth_rate

from apps.tuberculosis.model import preprocess, outputs
from apps.tuberculosis.model.validate import validate_params
from apps.tuberculosis.model.stratification import stratify_by_organ, stratify_by_age, apply_universal_stratification


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a simple SIR model
    """
    validate_params(params)

    # Define model times.
    integration_times = get_model_times_from_inputs(
        round(params["start_time"]), params["end_time"], params["time_step"]
    )

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

    # Define initial conditions - 1 infectious person.
    init_conditions = {Compartment.INFECTIOUS: 1}

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

    # Set some parameter values or parameters that require pre-processing
    params, treatment_death_func, relapse_func = preprocess.flows.process_unstratified_parameter_values(params)

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
        starting_population=params['start_population_size'],
    )

    # Apply infectiousness adjustment for individuals on treatment
    tb_model.individual_infectiousness_adjustments = treatment_infectiousness_adjustment

    # apply age stratification
    if "age" in params["stratify_by"]:
        stratify_by_age(tb_model, params, compartments)
    else:
        # set time-variant functions for treatment death and relapse rates
        tb_model.time_variants['treatment_death_rate'] = treatment_death_func
        tb_model.time_variants['relapse_rate'] = relapse_func

    # apply user-defined universal stratifications
    for stratification in [s for s in params["stratify_by"] if s[:10] == "universal_"]:
        stratification_details = params['universal_stratifications'][stratification]
        apply_universal_stratification(tb_model, compartments, stratification, stratification_details)

    if "organ" in params["stratify_by"]:
        stratify_by_organ(tb_model, params)

    # Load time-variant birth rates
    set_model_time_variant_birth_rate(tb_model, params['iso3'])

    # Register derived output functions
    # These functions calculate 'derived' outputs of interest, based on the
    # model's compartment values or flows. These are calculated after the model is run.
    func_outputs = {
        "prevalence_infectious": outputs.calculate_prevalence_infectious,
        "prevalence_susceptible": outputs.calculate_prevalence_susceptible,
        "population_size": outputs.calculate_population_size,
    }
    tb_model.add_function_derived_outputs(func_outputs)

    return tb_model
