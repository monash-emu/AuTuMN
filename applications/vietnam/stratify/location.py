import numpy as np
from summer_py.summer_model import StratifiedModel
from autumn.tb_model import scale_relative_risks_for_equivalence


def stratify_location(model: StratifiedModel, location_params: dict):
    strata = location_params["strata"]
    proportions = location_params["proportions"]
    raw_relative_risks_loc = location_params["relative_risk_transmission"]
    scaled_relative_risks_loc = scale_relative_risks_for_equivalence(
        proportions, raw_relative_risks_loc
    )

    # Adjusted such that heterogeneous mixing yields similar overall burden as homogeneous
    location_mixing = 3.0 * np.array(location_params["mixing_matrix"])

    location_adjustments = {}
    location_adjustments["acf_rate"] = location_params["acf_rate"]
    for beta_type in ["", "_late_latent", "_recovered"]:
        location_adjustments[f"contact_rate{beta_type}"] = scaled_relative_risks_loc

    model.stratify(
        "location",
        ["rural_province", "urban_nonger", "urban_ger", "prison"],
        [],
        requested_proportions=proportions,
        entry_proportions=proportions,
        adjustment_requests=location_adjustments,
        mixing_matrix=location_mixing,
        verbose=False,
    )
