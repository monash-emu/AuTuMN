import os
import yaml
import nevergrad as ng
import pymc as pm
import datetime

from estival.optimization import nevergrad as eng
from estival.calibration import pymc as epm

from autumn.settings.folders import PROJECTS_PATH
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object

INCLUDED_COUNTRIES  = yaml.load(open(os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")), Loader=yaml.UnsafeLoader)

ANALYSES_NAMES = ["main", "no_google_mobility", "increased_hh_contacts"]

def optimise_model_fit(bcm, warmup_iterations: int = 2000, search_iterations: int = 5000):

    # Build optimizer
    opt = eng.optimize_model(bcm, obj_function=bcm.loglikelihood)

    # Run warm-up iterations and 
    res = opt.minimize(warmup_iterations)

    res = opt.minimize(search_iterations)
    best_params = res.value[1]

    # return optimal parameters and optimisation object in case we want to resume the search afterwards
    return best_params, opt


def resume_opti_search(opt, extra_iterations: int = 5000):

    res = opt.minimize(extra_iterations)
    best_params = res.value[1]
    
    return best_params, opt


def sample_with_pymc(bcm, initvals, draws=1000, tune=500, cores=8, chains=8):

    with pm.Model() as model:    
        variables = epm.use_model(bcm)
        idata = pm.sample(step=[pm.DEMetropolis(variables)], draws=draws, tune=tune, cores=cores,chains=chains, initvals=initvals)

    return idata

