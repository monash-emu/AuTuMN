# This is required for pymc parallel evaluation in notebooks

import multiprocessing as mp
import platform
import sys

from time import time

import summer2
import numpy as np
import pandas as pd
import pymc as pm

from estival.wrappers import pymc as epm

from summer2.extras import test_models

# Targets represent data we are trying to fit to
from estival import targets as est

# We specify parameters using (Bayesian) priors
from estival import priors as esp

# Finally we combine these with our summer2 model in a BayesianCompartmentalModel (BCM)
from estival.model import BayesianCompartmentalModel


def run_calibration():
    start = time()
    print(f"Starting {time()-start}", flush=True)
    m = test_models.sirs_parametric_age(times=[0, 100], agegroups=list(range(128)))
    defp = m.get_default_parameters()

    m.run({"contact_rate": 0.1, "recovery_rate": 0.4})
    do_def = m.get_derived_outputs_df()
    obs_clean = do_def["incidence"].iloc[0:50]
    obs_noisy = obs_clean * np.exp(np.random.normal(0.0, 0.2, len(obs_clean)))

    print(f"Building BCM {time()-start}", flush=True)
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

    print(f"Calibrating {time()-start}", flush=True)

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
            step=[pm.DEMetropolis(variables)],
            draws=10000,
            tune=0,
            cores=8,
            chains=8,
            progressbar=False,
        )

    print(f"Writing {time()-start}", flush=True)

    idata.to_netcdf("thing.nc")

    print(f"Finished {time()-start}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    run_calibration()
    sys.exit(0)
