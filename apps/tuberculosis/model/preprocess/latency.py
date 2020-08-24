

# get parameter values from Ragonnet et al., Epidemics 2017
def get_unstratified_parameter_values(params):
    params['stabilisation_rate'] = 365.25 * 1.0e-2
    params['early_activation_rate'] = 365.25 * 1.1e-3
    params['late_activation_rate'] = 365.25 * 5.5e-6

    return params

