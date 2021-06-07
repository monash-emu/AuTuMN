from summer import CompartmentalModel

from autumn.models.tuberculosis.parameters import Parameters
from autumn.tools.project import Params, build_rel_path
from autumn.tools.curve import scale_up_function, tanh_based_scaleup

from .constants import Compartment

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)

COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.EARLY_LATENT,
    Compartment.LATE_LATENT,
    Compartment.INFECTIOUS,
    Compartment.ON_TREATMENT,
    Compartment.RECOVERED,
]
INFECTIOUS_COMPS = [
    Compartment.INFECTIOUS,
    Compartment.ON_TREATMENT,
]


def build_model(params: dict) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)
    model = CompartmentalModel(
        # FIXME: critical_ranges missing
        times=[params.time.start, params.time.end],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=params.time.step,
    )

    # Add initial population
    init_pop = {
        Compartment.INFECTIOUS: 1,
        Compartment.SUSCEPTIBLE: params.start_population_size,
    }
    model.set_initial_population(init_pop)

    # FIXME: DO THIS NEXT
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

    # Infection flows.
    model.add_infection_frequency_flow(
        "infection", params.contact_rate, Compartment.SUSCEPTIBLE, Compartment.EARLY_LATENT
    )
    model.add_infection_frequency_flow(
        "infection_from_latent",
        params.contact_rate * params.rr_infection_latent,
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_recovered",
        params.contact_rate * params.rr_infection_recovered,
        Compartment.RECOVERED,
        Compartment.EARLY_LATENT,
    )

    # Transition flows.
    model.add_transition_flow(
        "stabilisation",
        params.stabilisation_rate,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
    )
    model.add_transition_flow(
        "early_activation",
        params.early_activation_rate,
        Compartment.EARLY_LATENT,
        Compartment.INFECTIOUS,
    )
    model.add_transition_flow(
        "late_activation",
        params.late_activation_rate,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
    )

    # Post-active-disease flows
    model.add_transition_flow(
        "self_recovery",
        params.self_recovery_rate_dict["unstratified"],
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    )

    # Detection rate for infected people.
    if "organ" in params.stratify_by:
        detection_rate = 1.0
    else:
        func_params = params.time_variant_tb_screening_rate
        screening_rate_func = tanh_based_scaleup(
            func_params["shape"],
            func_params["inflection_time"],
            func_params["lower_asymptote"],
            func_params["upper_asymptote"],
        )

        def detection_rate(t):
            return screening_rate_func(t) * params.passive_screening_sensitivity["unstratified"]

    model.add_transition_flow(
        "detection",
        detection_rate,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
    )

    # Treatment recovery, releapse, death flows.
    # Set unstratified treatment-outcome-related parameters
    if "age" in params["stratify_by"]:
        # Relapse and treatment death need to be adjusted by age later.
        treatment_recovery_rate = 1.0
        treatment_death_rate = 1.0
        relapse_rate = 1.0
    else:
        times = list(params.time_variant_tsr.keys())
        vals = list(params.time_variant_tsr.values())
        time_variant_tsr = scale_up_function(times, vals, method=4)

        def treatment_recovery_rate(t):
            return max(
                1 / params.treatment_duration,
                params.universal_death_rate
                / params.prop_death_among_negative_tx_outcome
                * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
            )

        def treatment_death_rate(t):
            return (
                params.prop_death_among_negative_tx_outcome
                * treatment_recovery_rate(t)
                * (1.0 - time_variant_tsr(t))
                / time_variant_tsr(t)
                - params.universal_death_rate
            )

        def relapse_rate(t):
            return (treatment_death_rate(t) + params.universal_death_rate) * (
                1.0 / params.prop_death_among_negative_tx_outcome - 1.0
            )

    model.add_transition_flow(
        "treatment_recovery",
        treatment_recovery_rate,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    )
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        Compartment.ON_TREATMENT,
        Compartment.INFECTIOUS,
    )
    model.add_death_flow(
        "treatment_death",
        treatment_death_rate,
        Compartment.ON_TREATMENT,
    )

    # Entry flows
    model.add_crude_birth_flow(
        "birth",
        params.crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    # Death flows
    if "age" in params["stratify_by"]:
        # If age-stratification is used, the baseline mortality rate is set to 1
        # so it can get multiplied by a time-variant
        universal_death_rate = 1.0
    else:
        universal_death_rate = params.universal_death_rate

    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)

    # Infection death
    model.add_death_flow(
        "infect_death",
        params.infect_death_rate_dict["unstratified"],
        Compartment.INFECTIOUS,
    )

    # Is active case finding (ACF) implemented?
    implement_acf = len(params.time_variant_acf) > 0
    if implement_acf:
        # Default value
        acf_detection_rate = 1.0

        # ACF flow parameter
        should_use_func = (
            len(params.time_variant_acf) == 1
            and params.time_variant_acf[0]["stratum_filter"] is None
        )
        if should_use_func:
            # Universal active case funding is applied
            times = list(params.time_variant_acf[0]["time_variant_screening_rate"].keys())
            vals = [
                v * params.acf_screening_sensitivity
                for v in list(params.time_variant_acf[0]["time_variant_screening_rate"].values())
            ]
            acf_detection_rate = scale_up_function(times, vals, method=4)

        model.add_transition_flow(
            "acf_detection",
            acf_detection_rate,
            Compartment.INFECTIOUS,
            Compartment.ON_TREATMENT,
        )

    # Is LTBI screening implemented?
    implement_ltbi_screening = len(params.time_variant_ltbi_screening) > 0
    if implement_ltbi_screening:
        # Default
        preventive_treatment_rate = 1.0
        should_use_func = (
            len(params.time_variant_ltbi_screening) == 1
            and params.time_variant_ltbi_screening[0]["stratum_filter"] is None
        )
        if should_use_func:
            # universal LTBI screening is applied
            times = list(
                params.time_variant_ltbi_screening[0]["time_variant_screening_rate"].keys()
            )
            vals = [
                v * params.ltbi_screening_sensitivity * params.pt_efficacy
                for v in list(
                    params.time_variant_ltbi_screening[0]["time_variant_screening_rate"].values()
                )
            ]
            preventive_treatment_rate = scale_up_function(times, vals, method=4)

        to_compartment_lookup = {
            "susceptible": Compartment.SUSCEPTIBLE,
            "recovered": Compartment.RECOVERED,
        }
        to_compartment = to_compartment_lookup[params.pt_destination_compartment]
        model.add_transition_flow(
            "preventive_treatment_early",
            preventive_treatment_rate,
            Compartment.EARLY_LATENT,
            to_compartment,
        )
        model.add_transition_flow(
            "preventive_treatment_late",
            preventive_treatment_rate,
            Compartment.LATE_LATENT,
            to_compartment,
        )

    # ==================================================

    # adjust late reactivation parameters using multiplier
    for latency_stage in ["early", "late"]:
        param_name = f"{latency_stage}_activation_rate"
        for key in params["age_specific_latency"][param_name]:
            params["age_specific_latency"][param_name][key] *= params["progression_multiplier"]

    # load unstratified latency parameters
    # get parameter values from Ragonnet et al., Epidemics 2017
    for param_name in ["stabilisation_rate", "early_activation_rate", "late_activation_rate"]:
        params[param_name] = (
            365.25 * params["age_specific_latency"][param_name]["unstratified"]
            if "age" not in params["stratify_by"]
            else 1.0
        )

    # =============================================

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

    # prepare infectiousness adjustment for individuals on treatment
    treatment_infectiousness_adjustment = [
        {
            "comp_name": Compartment.ON_TREATMENT,
            "comp_strata": {},
            "value": params["on_treatment_infect_multiplier"],
        }
    ]

    # Apply infectiousness adjustment for individuals on treatment
    tb_model.individual_infectiousness_adjustments = treatment_infectiousness_adjustment
