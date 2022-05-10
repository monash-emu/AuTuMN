from autumn.settings import Region, Models
from autumn.tools.optimisation.opti_tools import get_calibration_object, calculate_objective_to_minimize

# Select the simulator
model, region = Models.COVID_19, Region.SRI_LANKA

# Or select a faster simulator for testing:
# model, region = Models.SM_SIR, Region.NCR

# Prepare the simulator
calib = get_calibration_object(model, region)
var_bounds = {
    prior['param_name']: prior['distri_params'] for prior in calib.all_priors
}

# Now everything is ready to go.
# The decision variables and their bounds can be found in var_bounds
# The objective function is called using calculate_objective_to_minimize
# See example below evaluating the objective at the variables' midpoints

params_dict = {
    param: (interval[0] + interval[1]) / 2. for param, interval in var_bounds.items()
}
y = calculate_objective_to_minimize(calib, params_dict)
# I obtained y = 12.463345415696706 on my laptop for the COVID_19 / SRI_LANKA model
# I obtained y = 229.01292047573264 on my laptop for the SM_SIR / NCR model
