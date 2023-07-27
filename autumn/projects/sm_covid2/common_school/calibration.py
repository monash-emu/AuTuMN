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


def make_rp_loglikelihood_func(len_rp_delta_values, rp_noise_sd):

    def rp_loglikelihood(params):
        sigma_square = rp_noise_sd ** 2
        sum_of_squares = jnp.sum(jnp.square(params['random_process.delta_values']))
        n = len_rp_delta_values
        
        return - n / 2. * jnp.log(2. * jnp.pi * sigma_square) - 1. / (2. * sigma_square) * sum_of_squares

    return rp_loglikelihood


def get_bcm_object(iso3, analysis="main"):

    assert analysis in ANALYSES_NAMES, "wrong analysis name requested"

    project = get_school_project(iso3, analysis)
    death_target_data = project.calibration.targets[0].data

    targets = [
        est.NegativeBinomialTarget(
            "infection_deaths", 
            death_target_data, 
            dispersion_param=esp.UniformPrior("infection_deaths_dispersion_param", (50, 200))
        )
    ]
    if len(project.calibration.targets) > 1:
        sero_target = project.calibration.targets[1]
        targets.append(
            est.BinomialTarget(
                "prop_ever_infected_age_matched", 
                sero_target.data, 
                sample_sizes=sero_target.sample_sizes
            )
        )

    # Add a safeguard target to prevent a premature epidemic occurring before the first reported death
    # Early calibrations sometimes produced a rapid epidemic reaching 100% attack rate before the true epidemic start
    zero_attack_rate_series = pd.Series([0.], index=[death_target_data.index[0]])
    targets.append(
        est.TruncatedNormalTarget(
            "prop_ever_infected",
            zero_attack_rate_series,
            trunc_range=[0., 0.10],  # will reject runs with >10% attack rate before epidemic
            stdev=1000.  # flat distribution
        )
    )

    default_configuration = project.param_set.baseline
    m = project.build_model(default_configuration.to_dict()) 

    priors = get_estival_uniform_priors(project.calibration.all_priors)

    default_params = m.builder.get_default_parameters()

    rp_ll = make_rp_loglikelihood_func(
        len(default_params['random_process.delta_values']), 
        project.param_set.baseline['random_process']['noise_sd']
    )
    bcm = BayesianCompartmentalModel(m, default_params, priors, targets, extra_ll=rp_ll)

    return bcm