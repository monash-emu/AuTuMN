import numpy as np
from autumn.runners.calibration.priors import UniformPrior, TruncNormalPrior

# Shared priors for mixing optimization calibration.
PRIORS = [
    UniformPrior("contact_rate", [0.03, 0.07]),
    UniformPrior("infectious_seed", [50, 600]),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=5.5,
        stdev=0.97,
        trunc_range=[1.0, np.inf],
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.5,
        stdev=0.77,
        trunc_range=[1.0, np.inf],
    ),
    # 3.8 to match the highest value found in Levin et al.
    UniformPrior("infection_fatality.multiplier", [0.5, 3.8]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.02, 0.20]),
    # Vary symptomatic and hospitalised proportions
    UniformPrior("clinical_stratification.props.symptomatic.multiplier", [0.6, 1.4]),
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.5, 1.5]),
    # Micro-distancing
    UniformPrior("mobility.microdistancing.behaviour.parameters.inflection_time", [60, 130]),
    UniformPrior("mobility.microdistancing.behaviour.parameters.end_asymptote", [0.25, 0.80]),
    # UniformPrior(
    #     "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time", [130, 250]
    # ),
    # UniformPrior(
    #     "mobility.microdistancing.behaviour_adjuster.parameters.start_asymptote", [0.4, 1.0]
    # ),
    # UniformPrior("elderly_mixing_reduction.relative_reduction", [0.0, 0.5]),
]
