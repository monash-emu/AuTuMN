

def get_latency_priors_from_epidemics(param_name, agegroup):
    lognormal_params = {
        'early_activation_rate': {
            'unstratified': [-6.78, 0.15],
            'age_0': [-5.00, 0.19],
            'age_5': [-5.91, 0.21],
            'age_15': [-8.05, 0.28],
        },
        'stabilisation_rate': {
            'unstratified': [-4.50, 0.13],
            'age_0': [-4.38, 0.19],
            'age_5': [-4.46, 0.18],
            'age_15': [-5.00, 0.28],
        },
        'late_activation_rate': {
            'unstratified': [-11.99, 0.34],
            'age_0': [-12.36, 1.13],
            'age_5': [-11.68, 0.67],
            'age_15': [-12.11, 0.45],
        }
    }

    prior = {
        "param_name": 'age_specific_latency.' + param_name + '.' + agegroup,
        "distribution": "lognormal",
        "distri_params": lognormal_params[param_name][agegroup],
    }
    return prior


def get_natural_history_priors_from_cid(param_name, organ):

    cid_estimates = {
        'infect_death_rate': {
            'smear_positive': [.335, .449],
            'smear_negative': [.017, .035],
        },
        'self_recovery_rate': {
            'smear_positive': [.177, .288],
            'smear_negative': [.073, .209]
        },
    }

    full_param_name = param_name + "_dict." + organ
    prior = {
        "param_name": full_param_name,
        "distribution": "uniform",
        "distri_params": cid_estimates[param_name][organ],
    }
    return prior
