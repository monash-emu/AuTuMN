"""
Tools to build a tuberculosis
"""

from .outputs import (
    load_calibration_from_db,
    store_tb_database,
    store_run_models,
    create_mcmc_outputs,
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
    plot_time_variant_param,
)
from .latency_params import provide_aggregated_latency_parameters, get_adapted_age_parameters
from .loaded_model import LoadedModel
from .interpolation import get_bcg_functions, add_birth_rate_functions
from .flows import (
    add_density_infection_flows,
    add_standard_infection_flows,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
)
from .preprocess import convert_competing_proportion_to_rate, scale_relative_risks_for_equivalence
