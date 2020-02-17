from summer_py.summer_model import StratifiedModel


def stratify_organ(model: StratifiedModel, organ_params: dict, detect_rate_by_organ: dict):
    strata = organ_params["strata"]
    props_smear = organ_params["proportions"]
    mortality_adjustments = organ_params["mortality_adjustments"]
    recovery_adjustments = organ_params["recovery_adjustments"]
    infectiousness_adjustments = organ_params["infectiousness_adjustments"]

    # workout the detection rate adjustment by organ status
    adjustment_smearneg = (
        detect_rate_by_organ["smearneg"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
        if detect_rate_by_organ["smearpos"](2015.0) > 0.0
        else 1.0
    )
    adjustment_extrapul = (
        detect_rate_by_organ["extrapul"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
        if detect_rate_by_organ["smearpos"](2015.0) > 0.0
        else 1.0
    )

    model.stratify(
        "organ",
        strata,
        ["infectious"],
        infectiousness_adjustments=infectiousness_adjustments,
        verbose=False,
        requested_proportions=props_smear,
        adjustment_requests={
            "recovery": recovery_adjustments,
            "infect_death": mortality_adjustments,
            "case_detection": {
                "smearpos": 1.0,
                "smearneg": adjustment_smearneg,
                "extrapul": adjustment_extrapul,
            },
            "early_progression": props_smear,
            "late_progression": props_smear,
        },
    )
