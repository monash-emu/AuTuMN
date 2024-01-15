from copy import copy 
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from jax import numpy as jnp

import summer2
from summer.utils import ref_times_to_dti

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel

from autumn.projects.sm_covid2.common_school.project_maker import get_school_project

ANALYSES_NAMES = ["main", "no_google_mobility", "increased_hh_contacts"]

def make_rp_loglikelihood_func(len_rp_delta_values, rp_noise_sd):

    def rp_loglikelihood(params):
        sigma_square = rp_noise_sd ** 2
        sum_of_squares = jnp.sum(jnp.square(params['random_process.delta_values']))
        n = len_rp_delta_values
        
        return - n / 2. * jnp.log(2. * jnp.pi * sigma_square) - 1. / (2. * sigma_square) * sum_of_squares

    return rp_loglikelihood


def get_bcm_object(iso3, analysis="main", scenario='baseline', _pymc_transform_eps_scale=.1):

    assert analysis in ANALYSES_NAMES, "wrong analysis name requested"
    assert scenario in ['baseline', 'scenario_1'], f"Requested scenario {scenario} not currently supported"
    
    project = get_school_project(iso3, analysis, scenario)
    death_target_data = project.death_target_data

    dispersion_prior = esp.UniformPrior("infection_deaths_dispersion_param", (200, 250))
    dispersion_prior._pymc_transform_eps_scale = _pymc_transform_eps_scale       

    targets = [
        est.NegativeBinomialTarget(
            "infection_deaths_ma7", 
            death_target_data, 
            dispersion_param=dispersion_prior
        )
    ]
    if project.sero_target is not None:
        targets.append(
            project.sero_target
        )

    # Add a safeguard target to prevent a premature epidemic occurring before the first reported death
    # Early calibrations sometimes produced a rapid epidemic reaching 100% attack rate before the true epidemic start
    zero_series = pd.Series([0.], index=[death_target_data.index[0]]) #  could be any value, only the time index matters
    
    def censored_func(modelled, data, parameters, time_weights):     
        # Returns a very large negative number if modelled value is greater than 1%. Returns 0 otherwise.
        return jnp.where(modelled > 0.01, -1.e11, 0.)[0]

    targets.append(
        est.CustomTarget(
            "prop_ever_infected", 
            zero_series, # could be any value, only the time index matters
            censored_func
        )
    )

    default_configuration = project.param_set.baseline
    m = project.build_model(default_configuration.to_dict()) 

    #priors = get_estival_uniform_priors(project.priors, _pymc_transform_eps_scale)

    default_params = m.builder.get_default_parameters()

    rp_ll = make_rp_loglikelihood_func(
        len(default_params['random_process.delta_values']), 
        project.param_set.baseline['random_process']['noise_sd']
    )
    bcm = BayesianCompartmentalModel(m, default_params, project.priors, targets, extra_ll=rp_ll)

    return bcm