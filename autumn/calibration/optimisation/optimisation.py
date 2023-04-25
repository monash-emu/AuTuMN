from autumn.calibration.optimisation.pyswarm_source import pso
from autumn.calibration.calibration import get_parameter_bounds_from_priors


def get_calibration_object(project):
    """
    Initialises a calibration object for the selected project.
    Also runs a quick calibration in order to:
    1. set some attributes required to evaluate the likelihood and prior
    2. perform a first (expensive) model run
    """
    calib = project.calibration
    
    # Quick run to finalise calibration object's initialisation
    calib.run(project,1,1,1)

    return calib


def calculate_objective_to_minimize(calib, params_dict):
    loglikelihood = calib.loglikelihood(params_dict)
    logprior = calib.logprior(params_dict)

    return - (loglikelihood + logprior)


def optimise_posterior_with_pso(project, n_particles, max_iterations):
    """
    Performs an optimisation of the posterior likelihood using PSO.

    Args:
        project: project
        n_particle: swarm size
        max_iterations: max number of PSO iterations

    Returns:
        Best solution as a dictionary of parameters 
    """
    calib = get_calibration_object(project)

    # Create a list of decision variables' names and the associated bounds 
    var_list = [prior['param_name'] for prior in calib.all_priors]
    var_bounds_list = [get_parameter_bounds_from_priors(prior) for prior in calib.all_priors]
    # var_bounds_list = [prior['distri_params'] for prior in calib.all_priors]

    # Objective function
    def obj_func(param_list):    
        params_dict = {par_name: param_list[i] for i, par_name in enumerate(var_list)}
        return calculate_objective_to_minimize(calib, params_dict)
    
    # Bounds around parameters
    lb = [bounds[0] for bounds in var_bounds_list]
    ub = [bounds[1] for bounds in var_bounds_list]

    # Run the PSO
    xopt, _ = pso(obj_func, lb, ub, swarmsize=n_particles, maxiter=max_iterations)

    best_param_dict = {par_name: xopt[i] for i, par_name in enumerate(var_list)}
    return best_param_dict