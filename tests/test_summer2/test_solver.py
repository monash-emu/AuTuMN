import pytest
import numpy as np

from summer2.solver import (
    solve_ode,
    solve_with_euler,
    solve_with_rk4,
    solve_with_odeint,
    solve_with_ivp,
    SolverType,
)
from numpy.testing import assert_allclose, assert_array_equal


SOLVERS = (
    (solve_with_euler, {"step_size": 0.001}, SolverType.EULER),
    (solve_with_rk4, {"step_size": 0.5}, SolverType.RUNGE_KUTTA),
    (solve_with_odeint, {}, SolverType.ODE_INT),
    (solve_with_ivp, {}, SolverType.SOLVE_IVP),
)


@pytest.mark.parametrize("solver, args, solver_type", SOLVERS)
def test_solve_ode_linear_func(solver, args, solver_type):
    """
    Ensure solver methods can solve a linear function's ODE.

    y_0 = 2 * t
    y_1 = 1 * t

    dy_0/dt = 2
    dy_1/dt = 1
    """

    def ode_func(vals, time):
        return np.array([2, 1])

    values = np.array([0, 0])
    times = np.array([0, 1, 2])
    expected_outputs = np.array(
        [
            # t = 0
            [0, 0],
            # t = 1
            [2, 1],
            # t = 2
            [4, 2],
        ]
    )
    output_arr_1 = solver(ode_func, values, times, solver_args=args)
    assert_allclose(expected_outputs, output_arr_1, rtol=0, atol=1e-9)
    output_arr_2 = solve_ode(solver_type, ode_func, values, times, solver_args=args)
    assert_allclose(expected_outputs, output_arr_2, rtol=0, atol=1e-9)
    assert_array_equal(output_arr_1, output_arr_2)


@pytest.mark.parametrize("solver, args, solver_type", SOLVERS)
def test_solve_ode_quadratic_func(solver, args, solver_type):
    """
    Ensure solver method can solve a quadratic function's ODE.

    y_0 = t ** 2
    y_1 = t ** 2 + t
    y_2 = 2 * t ** 2

    dy_0/dt = 2 * t
    dy_1/dt = 2 * t + 1
    dy_2/dt = 4 * t
    """

    def ode_func(vals, time):
        return np.array([2 * time, 2 * time + 1, 4 * time])

    values = np.array([0, 1, 2])
    times = np.array([0, 1, 2, 3])
    expected_outputs = np.array(
        [
            # t = 0
            [0 ** 2, 0 ** 2 + 0 + 1, 2 * 0 ** 2 + 2],
            # t = 1
            [1 ** 2, 1 ** 2 + 1 + 1, 2 * 1 ** 2 + 2],
            # t = 2
            [2 ** 2, 2 ** 2 + 2 + 1, 2 * 2 ** 2 + 2],
            # t = 3
            [3 ** 2, 3 ** 2 + 3 + 1, 2 * 3 ** 2 + 2],
        ]
    )
    output_arr_1 = solver(ode_func, values, times, solver_args=args)
    assert_allclose(expected_outputs, output_arr_1, rtol=0, atol=1e-2)
    output_arr_2 = solve_ode(solver_type, ode_func, values, times, solver_args=args)
    assert_allclose(expected_outputs, output_arr_2, rtol=0, atol=1e-2)
    assert_array_equal(output_arr_1, output_arr_2)
