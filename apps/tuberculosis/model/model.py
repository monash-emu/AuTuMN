from copy import deepcopy

from summer.legacy.model import StratifiedModel
from summer.legacy.constants import Compartment, BirthApproach
from autumn.utils.scenarios import get_model_times_from_inputs
from autumn import inputs
from autumn.curve import scale_up_function
from apps.tuberculosis.model import preprocess, outputs
from apps.tuberculosis.model.validate import validate_params, check_param_values
from apps.tuberculosis.model.stratification import (
    stratify_by_organ,
    stratify_by_age,
    apply_user_defined_stratification,
)


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
