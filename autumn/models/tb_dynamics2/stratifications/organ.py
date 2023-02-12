from summer2 import Stratification, Multiply, Overwrite

from autumn.models.tb_dynamics2.constants import INFECTIOUS_COMPS, OrganStratum
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from summer2.parameters import Time, Function
#from operator import attrgetter


ORGAN_STRATA = [
    OrganStratum.SMEAR_POSITIVE,
    OrganStratum.SMEAR_NEGATIVE,
    OrganStratum.EXTRAPULMONARY,
]


def get_organ_strat(params) -> Stratification:
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
            getattr(params.infect_death_rate_dict, effective_stratum, None).value
        )

    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    #Define different natural history (self recovery) by organ status
    self_recovery_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        effective_stratum = (
            OrganStratum.SMEAR_NEGATIVE
            if organ_stratum == OrganStratum.EXTRAPULMONARY
            else organ_stratum
        )

        self_recovery_adjs[organ_stratum] = Overwrite(
            getattr(params.self_recovery_rate_dict, effective_stratum, None).value
        )

    strat.set_flow_adjustments("self_recovery", self_recovery_adjs)

    # # Define different detection rates by organ status.
    sensitivity = params.passive_screening_sensitivity
    screening_rate_func = build_static_sigmoidal_multicurve(list(params.time_variant_tb_screening_rate.keys()), 
                                                            list(params.time_variant_tb_screening_rate.values())
                                                            )
    detection_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        #adj_vals = sensitivity[organ_stratum]
        adj_vals = getattr(sensitivity, organ_stratum, 1.0)
        detection_adjs[organ_stratum] = params.cdr_adjustment * Function(detection_func,[Time, screening_rate_func, adj_vals])
        
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


def detection_func(t, tfunc, val):
    return tfunc(t) * val
