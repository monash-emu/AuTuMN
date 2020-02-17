from summer_py.summer_model import StratifiedModel


def stratify_strain(model: StratifiedModel, strain_params: dict):
    strata = strain_params["strata"]
    infectiousness_adjustments = strain_params["infectiousness_adjustments"]
    prop_mdr_detected_as_mdr = strain_params["prop_mdr_detected_as_mdr"]
    mdr_tsr = strain_params["mdr_tsr"]
    ipt_rate = strain_params["ipt_switch"]

    # Divide by .9 for last DS TSR
    mdr_adjustment = prop_mdr_detected_as_mdr * mdr_tsr / 0.9

    model.stratify(
        "strain",
        strata,
        ["early_latent", "late_latent", "infectious"],
        verbose=False,
        requested_proportions={"mdr": 0.0},
        adjustment_requests={
            "contact_rate": {"ds": 1.0, "mdr": 1.0},
            "case_detection": {"mdr": mdr_adjustment},
            "ipt_rate": ipt_rate,
        },
        infectiousness_adjustments=infectiousness_adjustments,
    )
