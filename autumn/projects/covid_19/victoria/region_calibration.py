from autumn.tools.calibration.priors import UniformPrior

# TODO: Get as much calibration code as possible into this file
# TODO: Check out vaccination and do visualisation for it in the inputs notebook
# TODO: Un-bury CDR function creation
# TODO: Look at contact tracing computed values
# TODO: Write visualisation notebook section for contact tracing
# TODO: Work out what is going on with seeding through importation

priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]
