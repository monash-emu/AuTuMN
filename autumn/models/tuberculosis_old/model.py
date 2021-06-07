import numpy as np
from copy import deepcopy

from summer.legacy.constants import BirthApproach, Compartment
from summer.legacy.model import StratifiedModel

from autumn.models.tuberculosis import outputs, preprocess
from autumn.models.tuberculosis.stratification import (
    apply_user_defined_stratification,
    stratify_by_age,
    stratify_by_organ,
)
from autumn.models.tuberculosis.validate import check_param_values, validate_params
from autumn.tools import inputs
from autumn.tools.curve import scale_up_function
from autumn.tools.project import Params, build_rel_path

base_params = Params(build_rel_path("params.yml"))


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a tuberculosis model
    """
    validate_params(params)  # perform validation of parameter format
    check_param_values(params)  # perform validation of some parameter values

    # Define model times.
    time = params["time"]
    integration_times = get_model_times_from_inputs(
        round(time["start"]), time["end"], time["step"], time["critical_ranges"]
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
    init_conditions = {
        Compartment.INFECTIOUS: 1,
    }

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

    # is ACF implemented?
    implement_acf = len(params["time_variant_acf"]) > 0
    if implement_acf:
        flows.append(preprocess.flows.ACF_FLOW)

    # is ltbi screening implemented?
    implement_ltbi_screening = len(params["time_variant_ltbi_screening"]) > 0
    if implement_ltbi_screening:
        flows += preprocess.flows.get_preventive_treatment_flows(
            params["pt_destination_compartment"]
        )

    # Set some parameter values or parameters that require pre-processing
    (
        params,
        treatment_recovery_func,
        treatment_death_func,
        relapse_func,
        detection_rate_func,
        acf_detection_rate_func,
        preventive_treatment_func,
        contact_rate_functions,
    ) = preprocess.flows.process_unstratified_parameter_values(
        params, implement_acf, implement_ltbi_screening
    )

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
        starting_population=int(params["start_population_size"]),
    )

    # register acf_detection_func
    if acf_detection_rate_func is not None:
        tb_model.time_variants["acf_detection_rate"] = acf_detection_rate_func
    # register preventive_treatment_func
    if preventive_treatment_func is not None:
        tb_model.time_variants["preventive_treatment_rate"] = preventive_treatment_func
    # register time-variant contact-rate functions:
    for param_name, func in contact_rate_functions.items():
        tb_model.time_variants[param_name] = func

    # Apply infectiousness adjustment for individuals on treatment
    tb_model.individual_infectiousness_adjustments = treatment_infectiousness_adjustment

    # apply age stratification
    if "age" in params["stratify_by"]:
        stratify_by_age(tb_model, params, compartments)
    else:
        # set time-variant functions for treatment death and relapse rates
        tb_model.time_variants["treatment_recovery_rate"] = treatment_recovery_func
        tb_model.time_variants["treatment_death_rate"] = treatment_death_func
        tb_model.time_variants["relapse_rate"] = relapse_func

    # Load time-variant birth rates
    birth_rates, years = inputs.get_crude_birth_rate(params["iso3"])
    birth_rates = [b / 1000.0 for b in birth_rates]  # birth rates are provided / 1000 population
    tb_model.time_variants["crude_birth_rate"] = scale_up_function(
        years, birth_rates, smoothness=0.2, method=5
    )
    tb_model.parameters["crude_birth_rate"] = "crude_birth_rate"

    # apply user-defined stratifications
    user_defined_stratifications = [
        s for s in list(params["user_defined_stratifications"].keys()) if s in params["stratify_by"]
    ]
    for stratification in user_defined_stratifications:
        assert "_" not in stratification, "Stratification name should not include '_'"
        stratification_details = params["user_defined_stratifications"][stratification]
        apply_user_defined_stratification(
            tb_model,
            compartments,
            stratification,
            stratification_details,
            implement_acf,
            implement_ltbi_screening,
        )

    if "organ" in params["stratify_by"]:
        stratify_by_organ(tb_model, params)
    else:
        tb_model.time_variants["detection_rate"] = detection_rate_func

    # Register derived output functions, which are calculations based on the model's compartment values or flows.
    # These are calculated after the model is run.
    outputs.get_all_derived_output_functions(
        params["calculated_outputs"], params["outputs_stratification"], tb_model
    )

    return tb_model


def get_model_times_from_inputs(start_time, end_time, time_step, critical_ranges=[]):
    """
    Find the time steps for model integration from the submitted requests, ensuring the time points are evenly spaced.
    Use a refined time-step within critical ranges
    """
    times = []
    interval_start = start_time
    for critical_range in critical_ranges:
        # add regularly-spaced points up until the start of the critical range
        interval_end = critical_range[0]
        if interval_end > interval_start:
            times += list(np.arange(interval_start, interval_end, time_step))
        # add points over the critical range with smaller time step
        interval_start = interval_end
        interval_end = critical_range[1]
        if interval_end > interval_start:
            times += list(np.arange(interval_start, interval_end, time_step / 10.0))
        interval_start = interval_end

    if end_time > interval_start:
        times += list(np.arange(interval_start, end_time, time_step))
    times.append(end_time)

    # clean up time values ending .9999999999
    times = [round(t, 5) for t in times]

    return np.array(times)
