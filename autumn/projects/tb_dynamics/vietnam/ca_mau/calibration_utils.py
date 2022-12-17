from autumn.calibration.priors import UniformPrior


CID_ESTIMATES = {
    "infect_death_rate": {
        "smear_positive": [0.335, 0.449],
        "smear_negative": [0.017, 0.035],
    },
    "self_recovery_rate": {"smear_positive": [0.177, 0.288], "smear_negative": [0.073, 0.209]},
}


def get_natural_history_priors_from_cid(param_name, organ):
    return UniformPrior(f"{param_name}_dict.{organ}", CID_ESTIMATES[param_name][organ])
