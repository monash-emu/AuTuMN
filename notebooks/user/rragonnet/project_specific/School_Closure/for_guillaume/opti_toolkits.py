from matplotlib import pyplot as plt
import datetime
from copy import copy
import numpy as np

from summer.utils import ref_times_to_dti

from autumn.core.project import get_project

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel


def initialise_bcm(region):
    project = get_project("sm_covid2", region)
    death_target = project.calibration.targets[0]
    default_configuration = project.param_set.baseline
    m = project.build_model(default_configuration.to_dict()) 

    targets = [
        # est.NegativeBinomialTarget(name="infection_deaths", data=death_target.data, dispersion_param=100.)
       est.NormalTarget(name="infection_deaths", data=death_target.data, stdev=100.)
    ]
    priors = get_estival_uniform_priors(project.calibration.all_priors)
    
    default_params = m.builder.get_default_parameters()
    bcm = BayesianCompartmentalModel(m, default_params, priors, targets)
    
    # Perform a first run so the following calls wiil be faster
    # mid_params = {name: (prior.bounds()[0] + prior.bounds()[1]) / 2. for name, prior in bcm.priors.items()}
    # bcm.run(mid_params)
    
    return bcm


def plot_fit(bcm, params):
    """
    Helper function to plot model fit to data
    """
    REF_DATE = datetime.date(2019,12,31)
    datetime_target = copy(bcm.targets["infection_deaths"].data)
    datetime_target.index = ref_times_to_dti(REF_DATE, datetime_target.index)

    ax = bcm.run(params).derived_outputs["infection_deaths"].plot()
    datetime_target.plot(style='.')
    ll = bcm.loglikelihood(**params)

    text = f"ll={round(ll, 4)}"
    plt.text(0.8, 0.9, text, transform=ax.transAxes)


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


def initialise_opti_pb(region):
    bcm = initialise_bcm(region)
    
    bounds = [prior.bounds() for _, prior in bcm.priors.items()]
    param_names = [name for name, _ in bcm.priors.items()]

    def objective_func(params_as_list):
        # objective to MINIMISE
        params_as_dict = params_flat_list_to_dict(params_as_list, param_names)
        return float(- bcm.loglikelihood(**params_as_dict))

    def plot_func(params_as_list):
        params_as_dict = params_flat_list_to_dict(params_as_list, param_names)
        plot_fit(bcm, params_as_dict)

    return objective_func, plot_func, bounds, param_names


def params_dict_to_flat_list(param_dict):
    return [v for k, v in param_dict.items() if k != 'random_process.delta_values'] + list(param_dict['random_process.delta_values'])


def params_flat_list_to_dict(param_list, param_names):

    param_dict = {}
    for i, name in enumerate(param_names):
        if name != "random_process.delta_values":
            param_dict[name] = param_list[i]
        else:
            param_dict[name] = np.array(
                param_list[i:]
            )
    return param_dict