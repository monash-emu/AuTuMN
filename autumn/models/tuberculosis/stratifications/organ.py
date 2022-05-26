from summer import Stratification, Multiply, Overwrite

from autumn.models.tuberculosis.constants import INFECTIOUS_COMPS, OrganStratum
from autumn.models.tuberculosis.parameters import Parameters
from autumn.model_features.curve.scale_up import make_linear_curve, tanh_based_scaleup

ORGAN_STRATA = [
    OrganStratum.SMEAR_POSITIVE,
    OrganStratum.SMEAR_NEGATIVE,
    OrganStratum.EXTRAPULMONARY,
]


def get_organ_strat(params: Parameters) -> Stratification:
    strat = Stratification("organ", ORGAN_STRATA, INFECTIOUS_COMPS)

    # Define infectiousness adjustment by organ status
    inf_adj = {}
    for stratum in ORGAN_STRATA:
        mult_key = f"{stratum}_infect_multiplier"
        inf_adj[stratum] = Multiply(getattr(params, mult_key, 1.0))

    for comp in INFECTIOUS_COMPS:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    # Define different natural history (infection death) by organ status
    infect_death_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        effective_stratum = (
            OrganStratum.SMEAR_NEGATIVE
            if organ_stratum == OrganStratum.EXTRAPULMONARY
            else organ_stratum
        )

        infect_death_adjs[organ_stratum] = Overwrite(
            params.infect_death_rate_dict[effective_stratum]
        )

    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    # Define different natural history (self recovery) by organ status
    self_recovery_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        effective_stratum = (
            OrganStratum.SMEAR_NEGATIVE
            if organ_stratum == OrganStratum.EXTRAPULMONARY
            else organ_stratum
        )

        self_recovery_adjs[organ_stratum] = Overwrite(
            params.self_recovery_rate_dict[effective_stratum]
        )

    strat.set_flow_adjustments("self_recovery", self_recovery_adjs)

    # Define different detection rates by organ status.
    screening_rate_func = tanh_based_scaleup(
        params.time_variant_tb_screening_rate["shape"],
        params.time_variant_tb_screening_rate["inflection_time"],
        params.time_variant_tb_screening_rate["end_asymptote"],
        params.time_variant_tb_screening_rate["start_asymptote"],
    )
    if params.awareness_raising:
        awaireness_linear_scaleup = make_linear_curve(
            x_0=params.awareness_raising["scale_up_range"][0],
            x_1=params.awareness_raising["scale_up_range"][1],
            y_0=1,
            y_1=params.awareness_raising["relative_screening_rate"],
        )

        def awareness_multiplier(t, cv):
            if t <= params.awareness_raising["scale_up_range"][0]:
                return 1.0
            elif t >= params.awareness_raising["scale_up_range"][1]:
                return params.awareness_raising["relative_screening_rate"]
            else:
                return awaireness_linear_scaleup(t)

    else:
        awareness_multiplier = lambda t, cv: 1.0

    combined_screening_rate_func = lambda t, cv: screening_rate_func(t, cv) * awareness_multiplier(t, cv)
    detection_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        detection_adjs[organ_stratum] = make_detection_func(
            organ_stratum, params, combined_screening_rate_func
        )

    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    strat.set_flow_adjustments("detection", detection_adjs)

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        OrganStratum.SMEAR_POSITIVE: params.incidence_props_pulmonary
        * params.incidence_props_smear_positive_among_pulmonary,
        OrganStratum.SMEAR_NEGATIVE: params.incidence_props_pulmonary
        * (1.0 - params.incidence_props_smear_positive_among_pulmonary),
        OrganStratum.EXTRAPULMONARY: 1.0 - params.incidence_props_pulmonary,
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)

    return strat


def make_detection_func(organ_stratum, params, screening_rate_func):
    def detection_func(t, cv):
        return screening_rate_func(t, cv) * params.passive_screening_sensitivity[organ_stratum]

    return detection_func
