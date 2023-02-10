from summer2 import Stratification, Multiply, Overwrite

from autumn.models.tuberculosis.constants import INFECTIOUS_COMPS, OrganStratum
from autumn.models.tuberculosis.parameters import Parameters
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from summer2.parameters import Time, Function


ORGAN_STRATA = [
    OrganStratum.SMEAR_POSITIVE,
    OrganStratum.SMEAR_NEGATIVE,
    OrganStratum.EXTRAPULMONARY,
]


def get_organ_strat(params: Parameters) -> Stratification:
    strat = Stratification("organ", ORGAN_STRATA, INFECTIOUS_COMPS)

    #Define infectiousness adjustment by organ status
    inf_adj = {}
    for stratum in ORGAN_STRATA:
        mult_key = f"{stratum}_infect_multiplier"
        inf_adj[stratum] = Multiply(getattr(params, mult_key, 1.0))

    for comp in INFECTIOUS_COMPS:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    #Define different natural history (infection death) by organ status
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
    screening_rate_func = build_static_sigmoidal_multicurve([k for k in params.time_variant_tb_screening_rate.keys()], [v for v in params.time_variant_tb_screening_rate.values()])
    detection_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        detection_adjs[organ_stratum] = Function(
             make_detection_func,
             [Time, screening_rate_func, organ_stratum, params]
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


def make_detection_func(t, screening_rate_func, organ_stratum, params):
    return screening_rate_func(t) * params.passive_screening_sensitivity[organ_stratum]
