from autumn.tools.calibration.priors import UniformPrior

# TODO: Get as much calibration code as possible into this file
# TODO: Look at contact tracing computed values

priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]
