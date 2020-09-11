from autumn.constants import Flow
from apps.tuberculosis.constants import Compartment
from .latency import get_unstratified_parameter_values
from autumn.curve import scale_up_function

DEFAULT_FLOWS = [
    # Infection flows.
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.SUSCEPTIBLE,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate",
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate_from_latent",
    },
    {
        "type": Flow.INFECTION_FREQUENCY,
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.EARLY_LATENT,
        "parameter": "contact_rate_from_recovered",
    },
    # Transition flows.
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.LATE_LATENT,
        "parameter": "stabilisation_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.LATE_LATENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "late_activation_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.EARLY_LATENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "early_activation_rate",
    },
    # Post-active-disease flows
    {
        "type": Flow.STANDARD,
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.RECOVERED,
        "parameter": "self_recovery_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.INFECTIOUS,
        "to": Compartment.ON_TREATMENT,
        "parameter": "detection_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.RECOVERED,
        "parameter": "treatment_recovery_rate",
    },
    {
        "type": Flow.STANDARD,
        "origin": Compartment.ON_TREATMENT,
        "to": Compartment.INFECTIOUS,
        "parameter": "relapse_rate",
    },
    # Infection death
    {"type": Flow.DEATH, "parameter": "infect_death_rate", "origin": Compartment.INFECTIOUS},
    {"type": Flow.DEATH, "parameter": "treatment_death_rate", "origin": Compartment.ON_TREATMENT},
]


def process_unstratified_parameter_values(params):
    """
    This function calculates some unstratified parameter values for parameters that need pre-processing. This usually
    involves combining multiple input parameters to determine a model parameter
    :return:
    """
    # Set unstratified detection flow parameter
    params['detection_rate'] = params['passive_screening_rate'] * params['passive_screening_sensitivity']['unstratified']

    # Set unstratified treatment-outcome-related parameters
    params['treatment_recovery_rate'] = 1 / params['treatment_duration']
    if "age" in params["stratify_by"]:  # relapse and treatment death need to be adjusted by age later
        params['treatment_death_rate'] = 1.
        params['relapse_rate'] = 1.
        treatment_death_func = None
        relapse_func = None
    else:
        input_time_variant_tsr = scale_up_function(
            list(params['time_variant_tsr'].keys()),
            list(params['time_variant_tsr'].values()),
            method=4,
        )
        effective_tsr_func = lambda t: min(
            input_time_variant_tsr(t),
            params['treatment_recovery_rate'] * params['prop_death_among_negative_tx_outcome'] /
            (params['treatment_recovery_rate'] * params['prop_death_among_negative_tx_outcome'] +
             params['universal_death_rate'])
        )

        def treatment_death_func(t):
            return params['prop_death_among_negative_tx_outcome'] * params['treatment_recovery_rate'] *\
                   (1. - effective_tsr_func(t)) / effective_tsr_func(t) - params['universal_death_rate']

        def relapse_func(t):
            return (treatment_death_func(t) + params['universal_death_rate']) *\
                   (1. / params['prop_death_among_negative_tx_outcome'] - 1.)

        params['treatment_death_rate'] = 'treatment_death_rate'
        params['relapse_rate'] = 'relapse_rate'

    # load latency parameters
    if params["override_latency_rates"]:
        params = get_unstratified_parameter_values(params)

    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
            params["contact_rate"] * params["rr_infection_" + state]
        )

    # assign unstratified parameter values to infection death and self-recovery processes
    for param_name in ["infect_death_rate", "self_recovery_rate"]:
        params[param_name] = params[param_name + "_dict"]["unstratified"]

    # if age-stratification is used, the baseline mortality rate is set to 1 so it can get multiplied by a time-variant
    if "age" in params["stratify_by"]:
        params['universal_death_rate'] = 1.

    return params, treatment_death_func, relapse_func
