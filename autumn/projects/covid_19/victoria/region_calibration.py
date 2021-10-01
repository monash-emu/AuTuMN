from autumn.tools.calibration.priors import UniformPrior

# TODO: Get as much calibration code as possible into this file
# TODO: Check out vaccination and do visualisation for it in the inputs notebook

priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]
