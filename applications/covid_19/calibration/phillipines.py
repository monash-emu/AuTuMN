from .base import run_calibration_chain, get_priors_and_targets

country = "philippines"
PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country, "deaths", 2)

# Get rid of time in the params to calibrate
del PAR_PRIORS[1]

target_to_plots = {
    "infection_deathXall": {
        "times": TARGET_OUTPUTS[0]["years"],
        "values": [[d] for d in TARGET_OUTPUTS[0]["values"]],
    }
}
print(target_to_plots)

run_calibration_chain(120, 1, country, PAR_PRIORS, TARGET_OUTPUTS)
