import numpy as np
from summer import CompartmentalModel

from autumn.models.tuberculosis.parameters import Parameters
from autumn.tools.project import Params, build_rel_path
from autumn.tools.curve import scale_up_function, tanh_based_scaleup
from autumn.tools import inputs
from autumn.tools.inputs.social_mixing.queries import get_prem_mixing_matrices
from autumn.tools.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices

from .constants import Compartment, COMPARTMENTS, INFECTIOUS_COMPS
from .stratifications.age import get_age_strat
from .stratifications.user_defined import get_user_defined_strat
from .stratifications.organ import get_organ_strat
from .outputs import request_outputs

base_params = Params(
    build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False
)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """
    params = Parameters(**params)
    time = params.time
    model = CompartmentalModel(
        times=[time.start, time.end],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time.step,
    )

    # Add initial population
    init_pop = {
        Compartment.INFECTIOUS: 1,
        Compartment.SUSCEPTIBLE: params.start_population_size - 1,
    }
    model.set_initial_population(init_pop)

    # build mixing matrix
    if params.age_mixing:
        if params.age_mixing.type == 'prem':
            age_mixing_matrices = get_prem_mixing_matrices(params.iso3, params.age_breakpoints, None)
        elif params.age_mixing.type == 'extrapolated':
            age_mixing_matrices = build_synthetic_matrices(
                params.iso3, params.age_mixing.source_iso3, params.age_breakpoints, params.age_mixing.age_adjust.bit_length,
                requested_locations=["all_locations"]
            )
        else:
            raise Exception("Invalid mixing matrix type specified in parameters")
    else:
        # Default to prem matrices (old model runs)
        age_mixing_matrices = get_prem_mixing_matrices(params.iso3, params.age_breakpoints, None)

    age_mixing_matrix = age_mixing_matrices["all_locations"]
    # convert daily contact rates to yearly rates
    age_mixing_matrix *= 365.25

    contact_rate = params.contact_rate
    contact_rate_latent = params.contact_rate * params.rr_infection_latent
    contact_rate_recovered = params.contact_rate * params.rr_infection_recovered
    if params.hh_contacts_pt:
        # PT in household contacts
        times = [params.hh_contacts_pt["start_time"], params.hh_contacts_pt["start_time"] + 1]
        vals = [0, params.hh_contacts_pt["prop_hh_contacts_screened"]]
        scaleup_screening_prop = scale_up_function(x=times, y=vals, method=4)

        def get_contact_rate_factory(contact_rate):
            def get_contact_rate(t, computed_values):
                rel_reduction = (
                    params.hh_contacts_pt["prop_smearpos_among_prev_tb"]
                    * params.hh_contacts_pt["prop_hh_transmission"]
                    * scaleup_screening_prop(t)
                    * params.ltbi_screening_sensitivity
                    * params.hh_contacts_pt["prop_pt_completion"]
                )
                return contact_rate * (1 - rel_reduction)

            return get_contact_rate

        contact_rate = get_contact_rate_factory(contact_rate)
        contact_rate_latent = get_contact_rate_factory(contact_rate_latent)
        contact_rate_recovered = get_contact_rate_factory(contact_rate_recovered)

    # Infection flows.
    model.add_infection_frequency_flow(
        "infection", contact_rate, Compartment.SUSCEPTIBLE, Compartment.EARLY_LATENT
    )
    model.add_infection_frequency_flow(
        "infection_from_latent",
        contact_rate_latent,
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_recovered",
        contact_rate_recovered,
        Compartment.RECOVERED,
        Compartment.EARLY_LATENT,
    )

    # Transition flows.
    stabilisation_rate = 1.0
    early_activation_rate = 1.0
    late_activation_rate = 1.0
    model.add_transition_flow(
        "stabilisation",
        stabilisation_rate,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
    )
    model.add_transition_flow(
        "early_activation",
        early_activation_rate,
        Compartment.EARLY_LATENT,
        Compartment.INFECTIOUS,
    )
    model.add_transition_flow(
        "late_activation",
        late_activation_rate,
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
            func_params["start_asymptote"],
            func_params["end_asymptote"],
        )

        def detection_rate(t, cv=None):
            return screening_rate_func(t, cv) * params.passive_screening_sensitivity["unstratified"]

    model.add_transition_flow(
        "detection",
        detection_rate,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
    )

    # Treatment recovery, releapse, death flows.
    # Relapse and treatment death need to be adjusted by age later.
    treatment_recovery_rate = 1.0
    treatment_death_rate = 1.0
    relapse_rate = 1.0
    model.add_transition_flow(
        "treatment_recovery",
        treatment_recovery_rate,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    )
    model.add_death_flow(
        "treatment_death",
        treatment_death_rate,
        Compartment.ON_TREATMENT,
    )
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        Compartment.ON_TREATMENT,
        Compartment.INFECTIOUS,
    )

    # Entry flows
    birth_rates, years = inputs.get_crude_birth_rate(params.iso3)
    birth_rates = [b / 1000.0 for b in birth_rates]  # Birth rates are provided / 1000 population
    crude_birth_rate = scale_up_function(years, birth_rates, smoothness=0.2, method=5)
    model.add_crude_birth_flow(
        "birth",
        crude_birth_rate,
        Compartment.SUSCEPTIBLE,
    )

    # Death flows - later modified by age stratification
    universal_death_rate = 1.0
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

    # Age stratification.
    age_strat = get_age_strat(params, age_mixing_matrix)
    model.stratify_with(age_strat)

    # Custom, user-defined stratifications
    user_defined_strats = [
        s for s in params.user_defined_stratifications.keys() if s in params.stratify_by
    ]
    for strat_name in user_defined_strats:
        assert "_" not in strat_name, "Stratification name should not include '_'"
        strat_details = params.user_defined_stratifications[strat_name]
        user_defined_strat = get_user_defined_strat(strat_name, strat_details, params)
        model.stratify_with(user_defined_strat)

    if "location" in params.user_defined_stratifications:
        location_strata = params.user_defined_stratifications["location"]["strata"]
    else:
        location_strata = []

    # Organ stratifications
    if "organ" in params.stratify_by:
        organ_strat = get_organ_strat(params)
        model.stratify_with(organ_strat)


    # Add importation flows
    if params.import_ltbi_cases:
        prop_early = 0.10  # assuming 10% of migrants with LTBI have been infected in the 6 months preceding their arrival.

        dest_cpt_fractions = {
            Compartment.EARLY_LATENT: prop_early,
            Compartment.LATE_LATENT: 1. - prop_early,
        }

        for dest_compartment, fraction in dest_cpt_fractions.items():
            # we need to account for the fact that the flows will be replicated n_dest_compartment times
            n_dest_compartment = len(model._compartment_name_map[dest_compartment])
            rate = fraction * params.import_ltbi_cases['n_cases_per_year'] / n_dest_compartment
            case_import_func = make_case_import_func(rate, params.import_ltbi_cases['start_time'])

            model.add_importation_flow(
                f"ltbi_importation_{dest_compartment}",
                case_import_func,
                dest=dest_compartment,
                split_imports=False
            )

    # Derived outputs
    request_outputs(
        model,
        params.cumulative_output_start_time,
        location_strata,
        params.time_variant_tb_screening_rate,
        implement_acf,
        implement_ltbi_screening,
        params.pt_efficacy,
        params.pt_sae_prop
    )

    return model


def make_case_import_func(rate, start_time):

    def case_import(time, computed_values):
        return rate if time >= start_time else 0.

    return case_import
