"""
The main entry point into using Jax in AuTuMN
This should imported before any model code that uses jax, or builds jax functions

The current jax status is stored in the environment variable AUTUMN_USE_JAX
Other modules should interact with this solely via set_using_jax and get_using_jax
"""


import warnings
import os

from typing import Union

import numpy as np
import scipy

try:
    from jax import numpy as jnp
    from jax import scipy as jsp
    from jax.config import config as _jax_config

    # Jax configuration
    # FIXME: We need to find a more appropriate place to ensure this happens globally
    _jax_config.update("jax_platform_name", "cpu")
    _jax_config.update("jax_enable_x64", True)
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

    Array = Union[jnp.ndarray, np.ndarray]

except ImportError:
    warnings.warn("Could not import jax, model performance will be limited")

    Array = np.ndarray


def set_using_jax(use_jax: bool):
    use_jax = "1" if use_jax else "0"
    os.environ["AUTUMN_USE_JAX"] = use_jax


def get_using_jax():
    use_jax = os.environ.get("AUTUMN_USE_JAX")
    if use_jax:
        return use_jax == "1"
    return False


def get_modules() -> dict:
    """Get the active numpy and scipy implementations for either jax or cpython,
    depending on whether the environment variable AUTUMN_USE_JAX is set

    Returns:
        A dictionary of modules
    """
    use_jax = get_using_jax()
    if use_jax:
        return {"numpy": jnp, "scipy": jsp}
    else:
        return {"numpy": np, "scipy": scipy}
