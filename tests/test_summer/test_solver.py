import numpy as np

from summer.model.utils.solver import solve_with_euler, solve_with_rk4


def test_solve_with_rk4_linear_func():
    """
    Ensure Runge-Kutta 4 method can solve a linear function's ODE.

    y_0 = 2 * t
    y_1 = 1 * t

    dy_0/dt = 2
    dy_1/dt = 1
    """

    def ode_func(vals, time):
        return np.array([2, 1])

    values = np.array([0, 0])
    times = np.array([0, 1, 2])
    output_arr = solve_with_rk4(ode_func, values, times, solver_args={"step_size": 0.5})
    expected_outputs = [
        # t = 0
        [0, 0],
        # t = 1
        [2, 1],
        # t = 2
        [4, 2],
    ]
    equals_arr = np.array(expected_outputs) == output_arr
    assert equals_arr.all()


def test_solve_with_rk4_quadratic_func():
    """
    Ensure Runge-Kutta 4 method can solve a quadratic function's ODE.

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
    output_arr = solve_with_rk4(ode_func, values, times, solver_args={"step_size": 0.01})
    expected_outputs = [
        # t = 0
        [0 ** 2, 0 ** 2 + 0 + 1, 2 * 0 ** 2 + 2],
        # t = 1
        [1 ** 2, 1 ** 2 + 1 + 1, 2 * 1 ** 2 + 2],
        # t = 2
        [2 ** 2, 2 ** 2 + 2 + 1, 2 * 2 ** 2 + 2],
        # t = 3
        [3 ** 2, 3 ** 2 + 3 + 1, 2 * 3 ** 2 + 2],
    ]
    tolerance = 0.001
    equals_arr = np.array(expected_outputs) - output_arr < tolerance
    assert equals_arr.all()


def test_solve_with_euler_linear_func():
    """
    Ensure Euler method can solve a linear function's ODE.

    y_0 = 2 * t
    y_1 = 1 * t

    dy_0/dt = 2
    dy_1/dt = 1
    """

    def ode_func(vals, time):
        return np.array([2, 1])

    values = np.array([0, 0])
    times = np.array([0, 1, 2])
    output_arr = solve_with_euler(ode_func, values, times, solver_args={"step_size": 0.5})
    expected_outputs = [
        # t = 0
        [0, 0],
        # t = 1
        [2, 1],
        # t = 2
        [4, 2],
    ]
    equals_arr = np.array(expected_outputs) == output_arr
    assert equals_arr.all()


def test_solve_with_euler_quadratic_func():
    """
    Ensure Euler method can solve a quadratic function's ODE.

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
    output_arr = solve_with_euler(ode_func, values, times, solver_args={"step_size": 0.01})
    expected_outputs = [
        # t = 0
        [0 ** 2, 0 ** 2 + 0 + 1, 2 * 0 ** 2 + 2],
        # t = 1
        [1 ** 2, 1 ** 2 + 1 + 1, 2 * 1 ** 2 + 2],
        # t = 2
        [2 ** 2, 2 ** 2 + 2 + 1, 2 * 2 ** 2 + 2],
        # t = 3
        [3 ** 2, 3 ** 2 + 3 + 1, 2 * 3 ** 2 + 2],
    ]
    tolerance = 0.1
    equals_arr = np.array(expected_outputs) - output_arr < tolerance
    assert equals_arr.all()
