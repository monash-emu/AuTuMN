from copy import copy 
import datetime

import numpy as np
import pandas as pd

from jax import numpy as jnp

import summer2
from summer.utils import ref_times_to_dti

from autumn.core.project import get_project

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel


def get_estival_uniform_priors(autumn_priors):
    estival_priors = []
    for prior_dict in autumn_priors:
        assert prior_dict["distribution"] == "uniform", "Only uniform priors are currently supported"
        
        if not "random_process.delta_values" in prior_dict["param_name"] and not "dispersion_param" in prior_dict["param_name"]:
            estival_priors.append(
                esp.UniformPrior(prior_dict["param_name"], prior_dict["distri_params"]),
            )
            
    ndelta_values = len([prior_dict for prior_dict in autumn_priors if prior_dict["param_name"].startswith("random_process.delta_values")])
    
    estival_priors.append(esp.UniformPrior("random_process.delta_values", [-2.0,2.0], ndelta_values + 1))
    
    return estival_priors


def make_rp_loglikelihood_func(len_rp_delta_values):

    def rp_loglikelihood(params):
        sigma_square = params['random_process.noise_sd'] ** 2
        sum_of_squares = jnp.sum(jnp.square(params['random_process.delta_values']))
        n = len_rp_delta_values
        
        return - n / 2. * jnp.log(2. * jnp.pi * sigma_square) - 1. / (2. * sigma_square) * sum_of_squares

    return rp_loglikelihood

def get_bcm_object(region):
    project = get_project("sm_covid2", region)
    death_target, sero_target = project.calibration.targets

    default_configuration = project.param_set.baseline
    m = project.build_model(default_configuration.to_dict()) 

    targets = [
        est.NegativeBinomialTarget(
            "infection_deaths", 
            death_target.data, 
            dispersion_param=esp.UniformPrior("infection_deaths_dispersion_param", (50, 200))
        ),
        est.BinomialTarget(
            "prop_ever_infected_age_matched", 
            sero_target.data, 
            sample_sizes=sero_target.sample_sizes
        )
    ]

    priors = get_estival_uniform_priors(project.calibration.all_priors)

    default_params = m.builder.get_default_parameters()

    rp_ll = make_rp_loglikelihood_func(len(default_params['random_process.delta_values']))
    bcm = BayesianCompartmentalModel(m, default_params, priors, targets, extra_ll=rp_ll)

    return bcm