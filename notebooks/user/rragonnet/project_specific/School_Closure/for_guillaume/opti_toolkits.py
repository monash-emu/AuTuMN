from matplotlib import pyplot as plt
import datetime
from copy import copy

from summer.utils import ref_times_to_dti

from autumn.core.project import get_project

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel


def initialise_bcm(region):
    project = get_project("sm_covid2", region)
    death_target = project.calibration.targets[0]

    default_params = project.param_set.baseline
    m = project.build_model(default_params.to_dict()) 

    targets = [
        # est.NegativeBinomialTarget(name="infection_deaths", data=death_target.data, dispersion_param=100.)
       est.NormalTarget(name="infection_deaths", data=death_target.data, stdev=100.)
    ]
    priors = get_estival_uniform_priors(project.calibration.all_priors)

    bcm = BayesianCompartmentalModel(m, default_params.to_dict(), priors, targets)
    
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
        estival_priors.append(
            esp.UniformPrior(prior_dict["param_name"], prior_dict["distri_params"]),
        ) 
    return estival_priors


def initialise_opti_pb(region):
    bcm = initialise_bcm(region)
    
    bounds = [prior.bounds() for _, prior in bcm.priors.items()]
    param_names = [name for name, _ in bcm.priors.items()]

    def convert_param_list_to_dict(params_as_list):
        return {param_names[i]: val for i, val in enumerate(params_as_list)}

    def objective_func(params_as_list):
        # objective to MINIMISE
        params_as_dict = convert_param_list_to_dict(params_as_list)
        return float(- bcm.loglikelihood(**params_as_dict))

    def plot_func(params_as_list):
        params_as_dict = convert_param_list_to_dict(params_as_list)
        plot_fit(bcm, params_as_dict)

    return objective_func, plot_func, bounds, param_names