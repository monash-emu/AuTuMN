# This is required for pymc parallel evaluation in notebooks

import multiprocessing as mp
import platform

if platform.system() != "Windows":
    mp.set_start_method("forkserver")

print(mp.cpu_count())

from time import time

import summer2
import numpy as np
import pandas as pd
import pymc as pm

from estival.wrappers import pymc as epm

from summer2.extras import test_models

start = time()

m = test_models.sirs_parametric_age(times=[0, 300], agegroups=list(range(240)))
defp = m.get_default_parameters()

m.run({"contact_rate": 0.1, "recovery_rate": 0.4})
do_def = m.get_derived_outputs_df()
obs_clean = do_def["incidence"].iloc[0:50]
obs_noisy = obs_clean * np.exp(np.random.normal(0.0, 0.2, len(obs_clean)))

# Targets represent data we are trying to fit to
from estival import targets as est

# We specify parameters using (Bayesian) priors
from estival import priors as esp

# Finally we combine these with our summer2 model in a BayesianCompartmentalModel (BCM)
from estival.model import BayesianCompartmentalModel

# Specify a Truncated normal target with a free dispersion parameter
targets = [
    est.TruncatedNormalTarget(
        "incidence",
        obs_noisy,
        (0.0, np.inf),
        esp.UniformPrior("incidence_dispersion", (0.1, obs_noisy.max() * 0.1)),
    )
]

# Uniform priors over our 2 model parameters
priors = [
    esp.UniformPrior("contact_rate", (0.01, 1.0)),
    esp.TruncNormalPrior("recovery_rate", 0.5, 0.2, (0.01, 1.0)),
]

bcm = BayesianCompartmentalModel(m, defp, priors, targets)

with pm.Model() as model:
    # This is all you need - a single call to use_model
    variables = epm.use_model(bcm)

    # The log-posterior value can also be output, but may incur additional overhead
    # Use jacobian=False to get the unwarped value (ie just the 'native' density of the priors
    # without transformation correction factors)
    # pm.Deterministic("logp", model.logp(jacobian=False))

    # Now call a sampler using the variables from use_model
    # In this case we use the Differential Evolution Metropolis sampler
    # See the PyMC docs for more details
    idata = pm.sample(
        step=[pm.DEMetropolis(variables)], draws=1000, tune=0, cores=8, chains=8, progressbar=False
    )
