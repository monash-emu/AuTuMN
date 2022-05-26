from autumn.calibration.priors import UniformPrior

# Not currently used anywhere.
LOGNORMAL_PARAMS = {
    "early_activation_rate": {
        "unstratified": [-6.78, 0.15],
        "age_0": [-5.00, 0.19],
        "age_5": [-5.91, 0.21],
        "age_15": [-8.05, 0.28],
    },
    "stabilisation_rate": {
        "unstratified": [-4.50, 0.13],
        "age_0": [-4.38, 0.19],
        "age_5": [-4.46, 0.18],
        "age_15": [-5.00, 0.28],
    },
    "late_activation_rate": {
        "unstratified": [-11.99, 0.34],
        "age_0": [-12.36, 1.13],
        "age_5": [-11.68, 0.67],
        "age_15": [-12.11, 0.45],
    },
}

CID_ESTIMATES = {
    "infect_death_rate": {
        "smear_positive": [0.335, 0.449],
        "smear_negative": [0.017, 0.035],
    },
    "self_recovery_rate": {"smear_positive": [0.177, 0.288], "smear_negative": [0.073, 0.209]},
}


def get_natural_history_priors_from_cid(param_name, organ):
    return UniformPrior(f"{param_name}_dict.{organ}", CID_ESTIMATES[param_name][organ])
