"""
Tools for solving compartmental ODEs
"""
from typing import Callable

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d


OdeFunction = Callable[[np.ndarray, float], np.ndarray]


class SolverType:
    """
    Options for ODE solver used by model
    """

    ODE_INT = "odeint"
    SOLVE_IVP = "solve_ivp"
    EULER = "euler"
    RUNGE_KUTTA = "rk4"


def solve_ode(
    solver_type: str,
    ode_func: OdeFunction,
    values: np.ndarray,
    times: np.ndarray,
    solver_args: dict,
) -> np.ndarray:
    """
    Solve an ODE function given a function describing the dynamics, some initial conditions and times.
    """
    if solver_type == SolverType.ODE_INT:
        return solve_with_odeint(ode_func, values, times, solver_args)
    elif solver_type == SolverType.SOLVE_IVP:
        return solve_with_ivp(ode_func, values, times, solver_args)
    elif solver_type == SolverType.EULER:
        return solve_with_euler(ode_func, values, times, solver_args)
    elif solver_type == SolverType.RUNGE_KUTTA:
        return solve_with_rk4(ode_func, values, times, solver_args)
    else:
        raise ValueError("Solver type requested is not available")


def solve_with_odeint(
    ode_func: OdeFunction, values: np.ndarray, times: np.ndarray, solver_args: dict
):
    """
    Solve ODE with SciPy's odeint solver.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    """
    atol = solver_args.get("atol", 1e-3)
    rtol = solver_args.get("rtol", 1e-3)
    return odeint(ode_func, values, times, atol=atol, rtol=rtol)


def solve_with_ivp(ode_func: OdeFunction, values: np.ndarray, times: np.ndarray, solver_args: dict):
    """
    Solve ODE with SciPy's solve_ivp solver.
    This method allows us to set a stopping condition.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """
    stopping_tolerance = solver_args.get("stopping_tolerance", 1e-60)

    def _ode_func(time, values):
        """Reverse parameters"""
        return ode_func(values, time)

    def _get_stopping_conditions(time, values):
        flows = ode_func(values, time)
        return max(list(map(abs, flows))) - stopping_tolerance

    _get_stopping_conditions.terminal = True
    t_span = (times[0], times[-1])
    results = solve_ivp(_ode_func, t_span, values, t_eval=times)
    # FIXME: the command below would make the optimisation analyses crash because of the stopping conditions
    # results = solve_ivp(_ode_func, t_span, values, t_eval=times, events=_get_stopping_conditions)
    return results["y"].transpose()


def solve_with_euler(
    ode_func: OdeFunction, values: np.ndarray, times: np.ndarray, solver_args: dict
):
    """
    Solve ODE with a hand-rolled Euler's method implementation.

    `WARNING: This method is too innacurate to use for real applications.`
    """
    step_size = solver_args.get("step_size", 0.1)
    start_time = times[0]
    end_time = times[-1]
    time_span = end_time - start_time
    num_timesteps = int(time_span / step_size) + 1
    assert (
        num_timesteps == time_span / step_size + 1
    ), f"Step size {step_size} must be a factor of the time span {time_span}."
    integration_times = np.linspace(start_time, end_time, num_timesteps)
    results_arr = np.zeros([num_timesteps, len(values)])
    results_arr[0] = np.array(values)

    # Perform Euler's method integration
    for time_idx, time in enumerate(integration_times[:-1]):
        values_arr = results_arr[time_idx]
        gradient_arr = ode_func(values_arr, time)
        results_arr[time_idx + 1] = values_arr + step_size * gradient_arr

    return _interpolate_solver_results(results_arr, integration_times, times)


def solve_with_rk4(ode_func: OdeFunction, values: np.ndarray, times: np.ndarray, solver_args: dict):
    """
    Solve ODE with a hand-rolled Runge-Kutta 4 implementation.

    `WARNING: This method is too innacurate to use for real applications.`
    """
    step_size = solver_args.get("step_size", 0.1)
    start_time = times[0]
    end_time = times[-1]
    time_span = end_time - start_time
    num_timesteps = int(time_span / step_size) + 1
    assert (
        num_timesteps == time_span / step_size + 1
    ), f"Step size {step_size} must be a factor of the time span {time_span}."
    integration_times = np.linspace(start_time, end_time, num_timesteps)
    results_arr = np.zeros([num_timesteps, len(values)])
    results_arr[0] = np.array(values)

    # Perform Runge-Kutta 4 method integration
    for time_idx, time in enumerate(integration_times[:-1]):
        values_arr = results_arr[time_idx]
        k1 = step_size * ode_func(values_arr, time)
        k2 = step_size * ode_func(values_arr + k1 / 2, time + step_size / 2)
        k3 = step_size * ode_func(values_arr + k2 / 2, time + step_size / 2)
        k4 = step_size * ode_func(values_arr + k3, time + step_size)
        results_arr[time_idx + 1] = values_arr + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return _interpolate_solver_results(results_arr, integration_times, times)


def _interpolate_solver_results(results_arr, integration_times, requested_times):
    """
    Interpolate solver results into an output array that matches the requested times

    results_arr: Solver results, 2D Numpy array
    integration_times: Times used to get solver results
    requested_times: Times to interpolate
    """
    # Build a function to produce interpolated results
    solved_func = interp1d(integration_times, results_arr, axis=0)
    # Create output array to store values for requested times.
    output_arr = np.zeros([len(requested_times), results_arr.shape[1]])
    output_arr[0] = results_arr[0]
    # Populate output array with interpolated results
    num_times = len(requested_times)
    for time_idx in range(1, num_times):
        time = requested_times[time_idx]
        output_arr[time_idx] = solved_func(time)

    return output_arr
