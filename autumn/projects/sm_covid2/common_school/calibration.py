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

from autumn.core.project import get_project


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


def plot_model_fit(bcm, params, region):
    REF_DATE = datetime.date(2019,12,31)

    targets = {}
    for output in ["infection_deaths", "prop_ever_infected_age_matched"]:
        t = copy(bcm.targets[output].data)
        t.index = ref_times_to_dti(REF_DATE, t.index)
        targets[output] = t

    run_model = bcm.run(params)
    ll = bcm.loglikelihood(**params)  # not ideal...

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), height_ratios=(2., 1., 2.), sharex=True)
    fig.suptitle(region.title())
    death_ax, rp_ax, sero_ax = axs[0], axs[1], axs[2]

    # Deaths
    run_model.derived_outputs["infection_deaths"].plot(ax=death_ax, ylabel="COVID-19 deaths")
    targets["infection_deaths"].plot(style='.', ax=death_ax)
    plt.text(0.8, 0.9, f"ll={round(ll, 4)}", transform=death_ax.transAxes)

    # Random Process
    run_model.derived_outputs["transformed_random_process"].plot(ax=rp_ax, ylabel="Random Process")
    y_max = max(rp_ax.get_ylim()[1], 1.1)
    xmin, xmax = rp_ax.get_xlim()
    rp_ax.set_ylim(0, y_max)
    rp_ax.hlines(y=1., xmin=xmin, xmax=xmax, linestyle="dotted", color="grey")

    # Sero data
    run_model.derived_outputs["prop_ever_infected_age_matched"].plot(ax=sero_ax, ylabel="Prop. seropositive\n(age-matched)")
    # targets["prop_ever_infected_age_matched"].plot(style='.', ax=sero_ax)

    return fig