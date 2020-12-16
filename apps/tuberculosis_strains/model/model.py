from summer2 import (
    CompartmentalModel,
    AgeStratification,
    StrainStratification,
    Stratification,
    adjust,
)
from autumn.constants import Compartment
from autumn.inputs import get_population_by_agegroup, get_death_rates_by_agegroup
from autumn.inputs.social_mixing.queries import get_mixing_matrix_specific_agegroups
from autumn.curve import scale_up_function

from apps.tuberculosis_strains.model import outputs
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)


COMPARTMENTS = [
    Compartment.SUSCEPTIBLE,
    Compartment.EARLY_LATENT,
    Compartment.LATE_LATENT,
    Compartment.INFECTIOUS,
    Compartment.DETECTED,
    Compartment.ON_TREATMENT,
    Compartment.RECOVERED,
]
INFECTIOUS_COMPS = [
    Compartment.INFECTIOUS,
    Compartment.DETECTED,
    Compartment.ON_TREATMENT,
]
INFECTED_COMPS = [
    Compartment.EARLY_LATENT,
    Compartment.LATE_LATENT,
    Compartment.INFECTIOUS,
    Compartment.DETECTED,
    Compartment.ON_TREATMENT,
]


def build_model(params: dict) -> CompartmentalModel:
    time = params["time"]
    model = CompartmentalModel(
        times=[time["start"], time["end"]],
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPS,
        timestep=time["step"],
    )

    # Add initial population
    init_pop = {
        Compartment.EARLY_LATENT: params["initial_early_latent_population"],
        Compartment.LATE_LATENT: params["initial_late_latent_population"],
        Compartment.INFECTIOUS: params["initial_infectious_population"],
        Compartment.DETECTED: params["initial_detected_population"],
        Compartment.ON_TREATMENT: params["initial_on_treatment_population"],
        Compartment.RECOVERED: 0,
    }
    sum_init_pop = sum(init_pop.values())
    init_pop[Compartment.SUSCEPTIBLE] = params["start_population_size"] - sum_init_pop
    model.set_initial_population(init_pop)

    # Add inter-compartmental flows
    params = _get_derived_params(params)
    # Entry flows
    model.add_crude_birth_flow(
        "birth",
        params["crude_birth_rate"],
        Compartment.SUSCEPTIBLE,
    )
    # Infection flows.
    model.add_infection_frequency_flow(
        "infection",
        params["contact_rate"],
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_latent",
        params["contact_rate_from_latent"],
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
    )
    model.add_infection_frequency_flow(
        "infection_from_recovered",
        params["contact_rate_from_recovered"],
        Compartment.RECOVERED,
        Compartment.EARLY_LATENT,
    )

    # Transition flows.
    model.add_fractional_flow(
        "treatment_early",
        params["preventive_treatment_rate"],
        Compartment.EARLY_LATENT,
        Compartment.RECOVERED,
    )
    model.add_fractional_flow(
        "treatment_late",
        params["preventive_treatment_rate"],
        Compartment.LATE_LATENT,
        Compartment.RECOVERED,
    )
    model.add_fractional_flow(
        "stabilisation",
        params["stabilisation_rate"],
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
    )
    model.add_fractional_flow(
        "early_activation",
        params["early_activation_rate"],
        Compartment.EARLY_LATENT,
        Compartment.INFECTIOUS,
    )
    model.add_fractional_flow(
        "late_activation",
        params["late_activation_rate"],
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
    )

    # Post-active-disease flows
    model.add_fractional_flow(
        "detection",
        params["detection_rate"],
        Compartment.INFECTIOUS,
        Compartment.DETECTED,
    )
    model.add_fractional_flow(
        "treatment_commencement",
        params["treatment_commencement_rate"],
        Compartment.DETECTED,
        Compartment.ON_TREATMENT,
    )
    model.add_fractional_flow(
        "missed_to_active",
        params["missed_to_active_rate"],
        Compartment.DETECTED,
        Compartment.INFECTIOUS,
    )
    model.add_fractional_flow(
        "self_recovery_infectious",
        params["self_recovery_rate"],
        Compartment.INFECTIOUS,
        Compartment.LATE_LATENT,
    )
    model.add_fractional_flow(
        "self_recovery_detected",
        params["self_recovery_rate"],
        Compartment.DETECTED,
        Compartment.LATE_LATENT,
    )
    model.add_fractional_flow(
        "treatment_recovery",
        params["treatment_recovery_rate"],
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    )
    model.add_fractional_flow(
        "treatment_default",
        params["treatment_default_rate"],
        Compartment.ON_TREATMENT,
        Compartment.INFECTIOUS,
    )
    model.add_fractional_flow(
        "failure_retreatment",
        params["failure_retreatment_rate"],
        Compartment.ON_TREATMENT,
        Compartment.DETECTED,
    )
    model.add_fractional_flow(
        "spontaneous_recovery",
        params["spontaneous_recovery_rate"],
        Compartment.ON_TREATMENT,
        Compartment.LATE_LATENT,
    )

    # Death flows
    # Universal death rate to be overriden by a multiply in age stratification.
    uni_death_flow_names = model.add_universal_death_flows("universal_death", death_rate=1)
    model.add_death_flow(
        "infectious_death",
        params["infect_death_rate"],
        Compartment.INFECTIOUS,
    )
    model.add_death_flow(
        "detected_death",
        params["infect_death_rate"],
        Compartment.DETECTED,
    )
    model.add_death_flow(
        "treatment_death",
        params["treatment_death_rate"],
        Compartment.ON_TREATMENT,
    )

    # Apply age-stratification
    age_strat = _build_age_strat(params, uni_death_flow_names)
    model.stratify_with(age_strat)

    # Add vaccination stratification.
    vac_strat = _build_vac_strat(params)
    model.stratify_with(vac_strat)

    # Apply organ stratification
    organ_strat = _build_organ_strat(params)
    model.stratify_with(organ_strat)

    # Apply strain stratification
    strain_strat = _build_strain_strat(params)
    model.stratify_with(strain_strat)

    # Add amplification flow
    model.add_fractional_flow(
        name="amplification",
        fractional_rate=params["amplification_rate"],
        source=Compartment.ON_TREATMENT,
        dest=Compartment.ON_TREATMENT,
        source_strata={"strain": "ds"},
        dest_strata={"strain": "mdr"},
        expected_flow_count=9,
    )

    # Add cross-strain reinfection flows
    model.add_infection_frequency_flow(
        name="reinfection_ds_to_mdr",
        contact_rate=params["reinfection_rate"],
        source=Compartment.EARLY_LATENT,
        dest=Compartment.EARLY_LATENT,
        source_strata={"strain": "ds"},
        dest_strata={"strain": "mdr"},
        expected_flow_count=3,
    )
    model.add_infection_frequency_flow(
        name="reinfection_mdr_to_ds",
        contact_rate=params["reinfection_rate"],
        source=Compartment.EARLY_LATENT,
        dest=Compartment.EARLY_LATENT,
        source_strata={"strain": "mdr"},
        dest_strata={"strain": "ds"},
        expected_flow_count=3,
    )

    model.add_infection_frequency_flow(
        name="reinfection_late_ds_to_mdr",
        contact_rate=params["reinfection_rate"],
        source=Compartment.LATE_LATENT,
        dest=Compartment.EARLY_LATENT,
        source_strata={"strain": "ds"},
        dest_strata={"strain": "mdr"},
        expected_flow_count=3,
    )
    model.add_infection_frequency_flow(
        name="reinfection_late_mdr_to_ds",
        contact_rate=params["reinfection_rate"],
        source=Compartment.LATE_LATENT,
        dest=Compartment.EARLY_LATENT,
        source_strata={"strain": "mdr"},
        dest_strata={"strain": "ds"},
        expected_flow_count=3,
    )

    # Apply classification stratification
    class_strat = _build_class_strat(params)
    model.stratify_with(class_strat)

    # Apply retention stratification
    retention_strat = _build_retention_strat(params)
    model.stratify_with(retention_strat)

    # # # Register derived output functions, which are calculations based on the model's compartment values or flows.
    # # # These are calculated after the model is run.
    # outputs.get_all_derived_output_functions(
    #     params["calculated_outputs"], params["outputs_stratification"], tb_model
    # )

    # flow_outputs = {}
    # flow_outputs["progression_early"] = TransitionFlowOutput(
    #     source=Compartment.EARLY_LATENT,
    #     dest=Compartment.INFECTIOUS,
    #     source_strata={},
    #     dest_strata={},
    # )
    # flow_outputs["progression_late"] = TransitionFlowOutput(
    #     source=Compartment.LATE_LATENT,
    #     dest=Compartment.INFECTIOUS,
    #     source_strata={},
    #     dest_strata={},
    # )
    # flow_outputs["notifications"] = TransitionFlowOutput(
    #     source=Compartment.INFECTIOUS,
    #     dest=Compartment.DETECTED,
    #     source_strata={},
    #     dest_strata={},
    # )
    # flow_outputs["infectious_deaths"] = InfectionDeathFlowOutput(
    #     source=Compartment.INFECTIOUS,
    #     source_strata={},
    # )
    # flow_outputs["detected_deaths"] = InfectionDeathFlowOutput(
    #     source=Compartment.DETECTED,
    #     source_strata={},
    # )
    # flow_outputs["treatment_deaths"] = InfectionDeathFlowOutput(
    #     source=Compartment.ON_TREATMENT,
    #     source_strata={},
    # )
    # tb_model.add_flow_derived_outputs(flow_outputs)

    # function_outputs = {}
    # function_outputs["progression"] = calculate_progression
    # function_outputs["prevalence_infectious"] = calculate_prevalence_infectious
    # function_outputs["percentage_latent"] = calculate_ltbi_percentage
    # function_outputs["disease_deaths"] = calculate_disease_deaths
    # function_outputs["incidence"] = calculate_disease_incidence

    # tb_model.add_function_derived_outputs(function_outputs)

    # TODO: Double check all strats aren't filtering
    return model


def _adjust_all_multiply(items: dict):
    return {s: adjust.Multiply(v) for s, v in items.items()}


def _build_age_strat(params: dict, uni_death_flow_names: list):
    # Apply age-stratification
    age_strat = AgeStratification("age", params["age_breakpoints"], COMPARTMENTS)

    # Set age demographics
    pop = get_population_by_agegroup(
        age_breakpoints=params["age_breakpoints"], country_iso_code=params["iso3"], year=2000
    )
    age_split_props = dict(zip(params["age_breakpoints"], [x / sum(pop) for x in pop]))
    age_strat.set_population_split(age_split_props)

    # Add age-based heterogeneous mixing
    mixing_matrix = get_mixing_matrix_specific_agegroups(
        country_iso_code=params["iso3"],
        requested_age_breaks=list(map(int, params["age_breakpoints"])),
        time_unit="years",
    )
    # FIXME: These values break the solver because they are very big.
    # age_strat.set_mixing_matrix(mixing_matrix)

    # Add age-based flow adjustments.
    age_strat.add_flow_adjustments(
        "stabilisation", _adjust_all_multiply(params["stabilisation_rate_stratified"]["age"])
    )
    age_strat.add_flow_adjustments(
        "early_activation", _adjust_all_multiply(params["early_activation_rate_stratified"]["age"])
    )
    age_strat.add_flow_adjustments(
        "late_activation", _adjust_all_multiply(params["late_activation_rate_stratified"]["age"])
    )

    # Add age-specific all-causes mortality rate.
    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(
        params["age_breakpoints"], params["iso3"]
    )

    death_rates_by_age = {
        age: scale_up_function(
            death_rate_years, death_rates_by_age[int(age)], smoothness=0.2, method=5
        )
        for age in params["age_breakpoints"]
    }
    for uni_death_flow_name in uni_death_flow_names:
        age_strat.add_flow_adjustments(
            uni_death_flow_name, _adjust_all_multiply(death_rates_by_age)
        )

    return age_strat


def _build_vac_strat(params):
    vac_strat = Stratification("vac", ["unvaccinated", "vaccinated"], [Compartment.SUSCEPTIBLE])
    vac_strat.set_population_split({"unvaccinated": 1.0, "vaccinated": 0.0})

    # Apply flow adjustments
    vac_strat.add_flow_adjustments(
        "infection",
        {
            "unvaccinated": adjust.Multiply(1.0),
            "vaccinated": adjust.Multiply(params["bcg"]["rr_infection_vaccinated"]),
        },
    )
    vac_strat.add_flow_adjustments(
        "treatment_early",
        {"unvaccinated": adjust.Multiply(0.0), "vaccinated": adjust.Multiply(1.0)},
    )
    vac_strat.add_flow_adjustments(
        "treatment_late", {"unvaccinated": adjust.Multiply(0.0), "vaccinated": adjust.Multiply(1.0)}
    )

    def time_varying_vaccination_coverage(t):
        return params["bcg"]["coverage"] if t > params["bcg"]["start_time"] else 0.0

    def time_varying_unvaccinated_coverage(t):
        return 1 - params["bcg"]["coverage"] if t > params["bcg"]["start_time"] else 1.0

    vac_strat.add_flow_adjustments(
        "birth",
        {
            "unvaccinated": adjust.Multiply(time_varying_unvaccinated_coverage),
            "vaccinated": adjust.Multiply(time_varying_vaccination_coverage),
        },
    )
    return vac_strat


def _build_organ_strat(params):
    organ_strat = Stratification(
        "organ", ["smear_positive", "smear_negative", "extra_pulmonary"], INFECTIOUS_COMPS
    )
    organ_strat.set_population_split(params["organ"]["props"])
    # Add infectiousness adjustments
    for comp in INFECTIOUS_COMPS:
        organ_strat.add_infectiousness_adjustments(
            comp, _adjust_all_multiply(params["organ"]["foi"])
        )

    organ_strat.add_flow_adjustments(
        "early_activation", _adjust_all_multiply(params["organ"]["props"])
    )
    organ_strat.add_flow_adjustments(
        "late_activation", _adjust_all_multiply(params["organ"]["props"])
    )
    organ_strat.add_flow_adjustments(
        "detection",
        _adjust_all_multiply(params["detection_rate_stratified"]["organ"]),
    )
    organ_strat.add_flow_adjustments(
        "self_recovery_infectious",
        _adjust_all_multiply(params["self_recovery_rate_stratified"]["organ"]),
    )
    organ_strat.add_flow_adjustments(
        "self_recovery_detected",
        _adjust_all_multiply(params["self_recovery_rate_stratified"]["organ"]),
    )
    return organ_strat


def _build_strain_strat(params):
    strat = StrainStratification("strain", ["ds", "mdr"], INFECTED_COMPS)
    strat.set_population_split({"ds": 0.5, "mdr": 0.5})
    for c in INFECTED_COMPS:
        strat.add_infectiousness_adjustments(
            c, {"ds": adjust.Multiply(1.0), "mdr": adjust.Multiply(0.8)}
        )

    strat.add_flow_adjustments(
        "treatment_commencement",
        _adjust_all_multiply(params["treatment_commencement_rate_stratified"]["strain"]),
    )
    treatment_adjusts = _adjust_all_multiply(
        params["preventive_treatment_rate_stratified"]["strain"]
    )
    strat.add_flow_adjustments("treatment_early", treatment_adjusts)
    strat.add_flow_adjustments("treatment_late", treatment_adjusts)
    strat.add_flow_adjustments(
        "treatment_recovery",
        _adjust_all_multiply(params["treatment_recovery_rate_stratified"]["strain"]),
    )
    strat.add_flow_adjustments(
        "treatment_death", _adjust_all_multiply(params["treatment_death_rate_stratified"]["strain"])
    )
    strat.add_flow_adjustments(
        "treatment_default",
        _adjust_all_multiply(params["treatment_default_rate_stratified"]["strain"]),
    )
    strat.add_flow_adjustments(
        "spontaneous_recovery",
        _adjust_all_multiply(params["spontaneous_recovery_rate_stratified"]["strain"]),
    )
    strat.add_flow_adjustments(
        "failure_retreatment",
        _adjust_all_multiply(params["failure_retreatment_rate_stratified"]["strain"]),
    )
    return strat


def _build_class_strat(params):
    strat = Stratification(
        "classified", ["correctly", "incorrectly"], [Compartment.DETECTED, Compartment.ON_TREATMENT]
    )
    # FIXME: Not sure how to represent these in SUMMERv2
    # strat.add_flow_adjustments("detection", {"correctly": adjust.Multiply(1.0), "incorrectly": adjust.Multiply(0.0)})
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "detection_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + organ
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_ds"
    #                 for age_group in params["age_breakpoints"]
    #                 for organ in organ_strata_requested
    #             ],
    #             [
    #                 {"correctly": 1.0, "incorrectly": 0.0}
    #                 for _ in range(len(params["age_breakpoints"] * len(organ_strata_requested)))
    #             ],
    #         )
    #     )
    # )
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "detection_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + organ
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_mdr"
    #                 for age_group in params["age_breakpoints"]
    #                 for organ in organ_strata_requested
    #             ],
    #             [
    #                 {
    #                     "correctly": params["frontline_xpert_prop"],
    #                     "incorrectly": 1.0 - params["frontline_xpert_prop"],
    #                 }
    #                 for _ in range(len(params["age_breakpoints"] * len(organ_strata_requested)))
    #             ],
    #         )
    #     )
    # )
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "detection_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + "extra_pulmonary"
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_mdr"
    #                 for age_group in params["age_breakpoints"]
    #             ],
    #             [
    #                 {"correctly": 0.0, "incorrectly": 1.0}
    #                 for _ in range(len(params["age_breakpoints"] * len(organ_strata_requested)))
    #             ],
    #         )
    #     )
    # )
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "spontaneous_recovery_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + organ
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_"
    #                 + strain
    #                 for age_group in params["age_breakpoints"]
    #                 for organ in organ_strata_requested
    #                 for strain in strain_strata_requested
    #             ],
    #             [
    #                 {
    #                     "correctly": params["spontaneous_recovery_rate_stratified"]["classified"][
    #                         "correctly"
    #                     ],
    #                     "incorrectly": params["spontaneous_recovery_rate_stratified"]["classified"][
    #                         "incorrectly"
    #                     ],
    #                 }
    #                 for _ in range(
    #                     len(params["age_breakpoints"])
    #                     * len(organ_strata_requested)
    #                     * len(strain_strata_requested)
    #                 )
    #             ],
    #         )
    #     )
    # )
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "failure_retreatment_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + organ
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_"
    #                 + strain
    #                 for age_group in params["age_breakpoints"]
    #                 for organ in organ_strata_requested
    #                 for strain in strain_strata_requested
    #             ],
    #             [
    #                 {
    #                     "correctly": params["failure_retreatment_rate_stratified"]["classified"][
    #                         "correctly"
    #                     ],
    #                     "incorrectly": params["failure_retreatment_rate_stratified"]["classified"][
    #                         "incorrectly"
    #                     ],
    #                 }
    #                 for _ in range(
    #                     len(params["age_breakpoints"])
    #                     * len(organ_strata_requested)
    #                     * len(strain_strata_requested)
    #                 )
    #             ],
    #         )
    #     )
    # )
    # classification_flow_adjustments.update(
    #     dict(
    #         zip(
    #             [
    #                 "treatment_default_rate"
    #                 + "X"
    #                 + age_stratification_name
    #                 + "_"
    #                 + age_group
    #                 + "X"
    #                 + organ_stratification_name
    #                 + "_"
    #                 + organ
    #                 + "X"
    #                 + strain_stratification_name
    #                 + "_mdr"
    #                 for age_group in params["age_breakpoints"]
    #                 for organ in organ_strata_requested
    #             ],
    #             [
    #                 {
    #                     "correctly": params["treatment_default_rate_stratified"]["classified"][
    #                         "correctly"
    #                     ],
    #                     "incorrectly": params["treatment_default_rate_stratified"]["classified"][
    #                         "incorrectly"
    #                     ],
    #                 }
    #                 for _ in range(
    #                     len(params["age_breakpoints"])
    #                     * len(organ_strata_requested)
    #                     * len(strain_strata_requested)
    #                 )
    #             ],
    #         )
    #     )
    # )
    return strat


def _build_retention_strat(params):
    strat = Stratification("retained", ["yes", "no"], [Compartment.DETECTED])
    strat.add_flow_adjustments(
        "detection",
        {
            "yes": adjust.Multiply(params["retention_prop"]),
            "no": adjust.Multiply(1 - params["retention_prop"]),
        },
    )
    strat.add_flow_adjustments(
        "missed_to_active",
        {"yes": adjust.Multiply(0), "no": adjust.Multiply(1)},
    )
    strat.add_flow_adjustments(
        "treatment_commencement",
        {"yes": adjust.Multiply(1), "no": adjust.Multiply(0)},
    )

    strat.add_flow_adjustments(
        "failure_retreatment",
        {"yes": adjust.Multiply(1), "no": adjust.Multiply(0)},
    )
    return strat


def _get_derived_params(params):
    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
            params["contact_rate"] * params["rr_infection_" + state]
        )

    params["detection_rate"] = (
        params["case_detection_prop_sp"]
        / (1 - params["case_detection_prop_sp"])
        * (
            params["self_recovery_rate"]
            * params["self_recovery_rate_stratified"]["organ"]["smear_positive"]
            + params["infect_death_rate"]
            * params["infect_death_rate_stratified"]["organ"]["smear_positive"]
        )
    )
    params["detection_rate_stratified"]["organ"]["smear_positive"] = 1.0
    params["detection_rate_stratified"]["organ"]["extra_pulmonary"] = (
        params["case_detection_prop_sp"]
        / (2 - params["case_detection_prop_sp"])
        * (
            params["infect_death_rate"]
            * params["infect_death_rate_stratified"]["organ"]["extra_pulmonary"]
            + params["self_recovery_rate"]
            * params["self_recovery_rate_stratified"]["organ"]["extra_pulmonary"]
        )
        / params["detection_rate"]
    )
    params["detection_rate_stratified"]["organ"]["smear_negative"] = (
        1
        / params["detection_rate"]
        * (
            params["detection_rate"]
            * params["detection_rate_stratified"]["organ"]["extra_pulmonary"]
            + (params["case_detection_prop"] - 0.5 * params["case_detection_prop_sp"])
            * params["frontline_xpert_prop"]
            * (
                params["infect_death_rate"]
                * params["infect_death_rate_stratified"]["organ"]["smear_negative"]
                + params["self_recovery_rate"]
                * params["self_recovery_rate_stratified"]["organ"]["smear_positive"]
            )
        )
    )

    params["treatment_recovery_rate"] = (
        params["treatment_success_prop"] / params["treatment_duration"]
    )
    params["treatment_recovery_rate_stratified"]["strain"]["ds"] = 1.0
    params["treatment_recovery_rate_stratified"]["strain"]["mdr"] = (
        params["treatment_success_prop"]
        * params["treatment_success_prop_stratified"]["strain"]["mdr"]
        / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"])
    ) / params["treatment_recovery_rate"]

    params["treatment_death_rate"] = (
        params["treatment_mortality_prop"] / params["treatment_duration"]
    )
    params["treatment_death_rate_stratified"]["strain"]["ds"] = 1.0
    params["treatment_death_rate_stratified"]["strain"]["mdr"] = (
        params["treatment_mortality_prop"]
        * params["treatment_mortality_prop_stratified"]["strain"]["mdr"]
        / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"])
    ) / params["treatment_death_rate"]

    params["treatment_default_rate"] = (
        params["treatment_default_prop"] / params["treatment_duration"]
    )
    params["treatment_default_rate_stratified"]["strain"]["ds"] = 1 - params["amplification_prob"]
    params["treatment_default_rate_stratified"]["strain"]["mdr"] = (
        params["treatment_default_prop"]
        * params["treatment_default_prop_stratified"]["strain"]["mdr"]
        / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"])
    ) / params["treatment_default_rate"]

    params["treatment_default_rate_stratified"]["classified"]["correctly"] = 1.0
    params["treatment_default_rate_stratified"]["classified"]["incorrectly"] = params[
        "proportion_mdr_misdiagnosed_as_ds_transition_to_fail_lost"
    ] / (
        params["treatment_default_rate"]
        * params["treatment_default_rate_stratified"]["strain"]["mdr"]
    )

    params["amplification_rate"] = params["treatment_default_rate"] * params["amplification_prob"]

    params["reinfection_rate"] = params["contact_rate"] * params["rr_infection_latent"]

    return params


def calculate_progression(
    time_idx,
    model,
    compartment_values,
    derived_outputs,
):
    return (
        derived_outputs["progression_early"][time_idx]
        + derived_outputs["progression_late"][time_idx]
    )


def calculate_disease_deaths(
    time_idx,
    model,
    compartment_values,
    derived_outputs,
):
    return (
        derived_outputs["infectious_deaths"][time_idx]
        + derived_outputs["detected_deaths"][time_idx]
        + derived_outputs["treatment_deaths"][time_idx]
    )


def calculate_prevalence_infectious(
    time_idx,
    model,
    compartment_values,
    derived_outputs,
):
    """
    Calculate the total number of infectious people at each time-step.
    """
    prevalence_infectious = 0
    for i, compartment in enumerate(model.compartment_names):
        is_infectious = compartment.has_name_in_list(
            [Compartment.INFECTIOUS, Compartment.DETECTED, Compartment.ON_TREATMENT]
        )
        if is_infectious:
            prevalence_infectious += compartment_values[i]
    pop_size = sum(compartment_values)

    return prevalence_infectious / pop_size * 1.0e5


def calculate_ltbi_percentage(
    time_idx,
    model,
    compartment_values,
    derived_outputs,
):
    """
    Calculate the total number of infectious people at each time-step.
    """
    prevalence_latent = 0
    for i, compartment in enumerate(model.compartment_names):
        is_latent = compartment.has_name_in_list(
            [Compartment.EARLY_LATENT, Compartment.LATE_LATENT]
        )
        if is_latent:
            prevalence_latent += compartment_values[i]
    pop_size = sum(compartment_values)

    return prevalence_latent / pop_size * 100


def calculate_disease_incidence(
    time_idx,
    model,
    compartment_values,
    derived_outputs,
):
    abs_incidence = (
        derived_outputs["progression_early"][time_idx]
        + derived_outputs["progression_late"][time_idx]
    )
    pop_size = sum(compartment_values)
    return abs_incidence / pop_size * 1.0e5
