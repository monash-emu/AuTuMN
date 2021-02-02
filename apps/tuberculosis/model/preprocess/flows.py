from summer.constants import Flow
from apps.tuberculosis.constants import Compartment
from .latency import get_unstratified_parameter_values
from autumn.curve import scale_up_function, tanh_based_scaleup


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
        "origin": Compartment.RECOVERED,
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

ACF_FLOW = {
    "type": Flow.STANDARD,
    "origin": Compartment.INFECTIOUS,
    "to": Compartment.ON_TREATMENT,
    "parameter": "acf_detection_rate",
}


def get_preventive_treatment_flows(destination_compartment):
    to_compartment = {"susceptible": Compartment.SUSCEPTIBLE, "recovered": Compartment.RECOVERED}
    preventive_treatment_flows = [
        {
            "type": Flow.STANDARD,
            "origin": Compartment.EARLY_LATENT,
            "to": to_compartment[destination_compartment],
            "parameter": "preventive_treatment_rate",
        },
        {
            "type": Flow.STANDARD,
            "origin": Compartment.LATE_LATENT,
            "to": to_compartment[destination_compartment],
            "parameter": "preventive_treatment_rate",
        },
    ]
    return preventive_treatment_flows


def process_unstratified_parameter_values(params, implement_acf, implement_ltbi_screening):
    """
    This function calculates some unstratified parameter values for parameters that need pre-processing. This usually
    involves combining multiple input parameters to determine a model parameter
    :return:
    """

    # Set unstratified detection flow parameter
    if "organ" in params["stratify_by"]:
        params["detection_rate"] = 1
        detection_rate_func = None
    else:
        screening_rate_func = tanh_based_scaleup(
            params["time_variant_tb_screening_rate"]["maximum_gradient"],
            params["time_variant_tb_screening_rate"]["max_change_time"],
            params["time_variant_tb_screening_rate"]["start_value"],
            params["time_variant_tb_screening_rate"]["end_value"],
        )

        def detection_rate_func(t):
            return screening_rate_func(t) * params["passive_screening_sensitivity"]["unstratified"]

        params["detection_rate"] = "detection_rate"

    # Set unstratified treatment-outcome-related parameters
    if (
        "age" in params["stratify_by"]
    ):  # relapse and treatment death need to be adjusted by age later
        params["treatment_recovery_rate"] = 1.0
        params["treatment_death_rate"] = 1.0
        params["relapse_rate"] = 1.0
        treatment_recovery_func = None
        treatment_death_func = None
        relapse_func = None
    else:
        time_variant_tsr = scale_up_function(
            list(params["time_variant_tsr"].keys()),
            list(params["time_variant_tsr"].values()),
            method=4,
        )

        def treatment_recovery_func(t):
            return max(
                1 / params["treatment_duration"],
                params["universal_death_rate"]
                / params["prop_death_among_negative_tx_outcome"]
                * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
            )

        def treatment_death_func(t):
            return (
                params["prop_death_among_negative_tx_outcome"]
                * treatment_recovery_func(t)
                * (1.0 - time_variant_tsr(t))
                / time_variant_tsr(t)
                - params["universal_death_rate"]
            )

        def relapse_func(t):
            return (treatment_death_func(t) + params["universal_death_rate"]) * (
                1.0 / params["prop_death_among_negative_tx_outcome"] - 1.0
            )

        params["treatment_recovery_rate"] = "treatment_recovery_rate"
        params["treatment_death_rate"] = "treatment_death_rate"
        params["relapse_rate"] = "relapse_rate"

    # adjust late reactivation parameters using multiplier
    for latency_stage in ["early", "late"]:
        param_name = f"{latency_stage}_activation_rate"
        for key in params["age_specific_latency"][param_name]:
            params["age_specific_latency"][param_name][key] *= params["progression_multiplier"]
    # load unstratified latency parameters
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
        params["universal_death_rate"] = 1.0

    # PT in household contacts
    contact_rate_functions = {}
    if params["hh_contacts_pt"]:
        scaleup_screening_prop = scale_up_function(
            x=[params["hh_contacts_pt"]["start_time"], params["hh_contacts_pt"]["start_time"] + 1],
            y=[0, params["hh_contacts_pt"]["prop_hh_contacts_screened"]],
            method=4,
        )

        def make_contact_rate_func(raw_contact_rate_value):
            def contact_rate_func(t):
                rel_reduction = (
                    params["hh_contacts_pt"]["prop_smearpos_among_prev_tb"]
                    * params["hh_contacts_pt"]["prop_hh_transmission"]
                    * scaleup_screening_prop(t)
                    * params["ltbi_screening_sensitivity"]
                    * params["hh_contacts_pt"]["prop_pt_completion"]
                )
                return raw_contact_rate_value * (1 - rel_reduction)

            return contact_rate_func

        for suffix in ["", "_from_latent", "_from_recovered"]:
            param_name = f"contact_rate{suffix}"
            raw_value = params[param_name]
            params[param_name] = param_name
            contact_rate_functions[param_name] = make_contact_rate_func(raw_value)

    # ACF flow parameter
    acf_detection_func = None
    if implement_acf:
        if (
            len(params["time_variant_acf"]) == 1
            and params["time_variant_acf"][0]["stratum_filter"] is None
        ):
            # universal ACF is applied
            acf_detection_func = scale_up_function(
                list(params["time_variant_acf"][0]["time_variant_screening_rate"].keys()),
                [
                    v * params["acf_screening_sensitivity"]
                    for v in list(
                        params["time_variant_acf"][0]["time_variant_screening_rate"].values()
                    )
                ],
                method=4,
            )
            params["acf_detection_rate"] = "acf_detection_rate"
        else:
            params["acf_detection_rate"] = 1.0

    # Preventive treatment flow parameters
    preventive_treatment_func = None
    if implement_ltbi_screening:
        if (
            len(params["time_variant_ltbi_screening"]) == 1
            and params["time_variant_ltbi_screening"][0]["stratum_filter"] is None
        ):
            # universal LTBI screening is applied
            preventive_treatment_func = scale_up_function(
                list(
                    params["time_variant_ltbi_screening"][0]["time_variant_screening_rate"].keys()
                ),
                [
                    v * params["ltbi_screening_sensitivity" * params["pt_efficacy"]]
                    for v in list(
                        params["time_variant_ltbi_screening"][0][
                            "time_variant_screening_rate"
                        ].values()
                    )
                ],
                method=4,
            )
            params["preventive_treatment_rate"] = "preventive_treatment_rate"
        else:
            params["preventive_treatment_rate"] = 1.0

    return (
        params,
        treatment_recovery_func,
        treatment_death_func,
        relapse_func,
        detection_rate_func,
        acf_detection_func,
        preventive_treatment_func,
        contact_rate_functions,
    )
