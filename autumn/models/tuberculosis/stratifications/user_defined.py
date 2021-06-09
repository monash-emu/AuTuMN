import numpy as np
from summer import Stratification, Multiply

from autumn.models.tuberculosis.constants import COMPARTMENTS
from autumn.models.tuberculosis.parameters import Parameters
from autumn.tools.curve import scale_up_function


def get_user_defined_strat(name: str, details: dict, params: Parameters) -> Stratification:
    """
    Stratify all model compartments based on a user-defined stratification request.
    """
    strat = Stratification(name, details["strata"], COMPARTMENTS)
    strat.set_population_split(details["proportions"])
    if "mixing_matrix" in details:
        mixing_matrix = np.array([row for row in details["mixing_matrix"]])
        strat.set_mixing_matrix(mixing_matrix)

    # Pre-process generic flow adjustments
    # IF infection is adjusted and other infection flows NOT adjusted
    # THEN use the infection adjustment for the other infection flows
    if "infection" in details["adjustments"]:
        for stage in ["latent", "recovered"]:
            flow_name = f"infection_from_{stage}"
            if flow_name not in details["adjustments"]:
                details["adjustments"][flow_name] = details["adjustments"]["infection"]

    # Adjust crude birth rate according to the strata proportions
    details["adjustments"]["birth"] = details["proportions"]

    # Set generic flow adjustments
    for flow_name, adjustment in details["adjustments"].items():
        adj = {k: Multiply(v) for k, v in adjustment.items()}
        strat.add_flow_adjustments(flow_name, adj)

    # ACF and preventive treatment interventions
    implement_acf = len(params.time_variant_acf) > 0
    implement_ltbi_screening = len(params.time_variant_ltbi_screening) > 0
    intervention_types = [
        # Active case finding.
        {
            "implement_switch": implement_acf,
            "flow_name": "acf_detection",
            "sensitivity": params.acf_screening_sensitivity,
            "prop_detected_effectively_moving": 1.0,
            "interventions": params.time_variant_acf,
        },
        # LTBI screening.
        {
            "implement_switch": implement_ltbi_screening,
            "flow_name": "preventive_treatment_early",
            "sensitivity": params.ltbi_screening_sensitivity,
            "prop_detected_effectively_moving": params.pt_efficacy,
            "interventions": params.time_variant_ltbi_screening,
        },
        {
            "implement_switch": implement_ltbi_screening,
            "flow_name": "preventive_treatment_late",
            "sensitivity": params.ltbi_screening_sensitivity,
            "prop_detected_effectively_moving": params.pt_efficacy,
            "interventions": params.time_variant_ltbi_screening,
        },
    ]

    for intervention_type in intervention_types:
        if not intervention_type["implement_switch"]:
            continue

        int_adjustments = {}
        for intervention in intervention_type["interventions"]:
            intervention_adj_func = make_intervention_adjustment_func(
                intervention["time_variant_screening_rate"],
                intervention_type["sensitivity"],
                intervention_type["prop_detected_effectively_moving"],
            )
            for stratum in details["strata"]:
                if stratum in intervention["stratum_filter"][name]:
                    int_adjustments[stratum] = intervention_adj_func
                else:
                    int_adjustments[stratum] = 0.0

            for age in params.age_breakpoints:
                should_exclude_age = age in intervention.get("exclude_age", [])
                age_int_adjs = {
                    k: Multiply(0) if should_exclude_age else Multiply(v)
                    for k, v in int_adjustments.items()
                }
                strat.add_flow_adjustments(
                    intervention_type["flow_name"], age_int_adjs, source_strata={"age": age}
                )

    return strat


def make_intervention_adjustment_func(
    time_variant_screening_rate, sensitivity, prop_detected_effectively_moving
):
    times = list(time_variant_screening_rate.keys())
    vals = [
        v * sensitivity * prop_detected_effectively_moving
        for v in list(time_variant_screening_rate.values())
    ]
    return scale_up_function(times, vals, method=4)
