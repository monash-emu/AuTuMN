import pytest
import numpy as np
from numpy.testing import assert_array_equal

from summer2 import CompartmentalModel, Compartment


def test_create_model():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert_array_equal(model.times, np.array([0, 1, 2, 3, 4, 5]))
    assert model.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    assert model._infectious_compartments == [Compartment("I")]
    assert_array_equal(model.initial_population, np.array([0, 0, 0]))

    # Times out of order
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=[5, 0], compartments=["S", "I", "R"], infectious_compartments=["I"]
        )

    # Start time negative
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=[-1, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
        )

    # Infectious compartment not a compartment
    with pytest.raises(AssertionError):
        CompartmentalModel(
            times=[-1, 5], compartments=["S", "I", "R"], infectious_compartments=["E"]
        )


def test_set_initial_population():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert_array_equal(model.initial_population, np.array([0, 0, 0]))
    model.set_initial_population({"S": 100})
    assert_array_equal(model.initial_population, np.array([100, 0, 0]))
    model.set_initial_population({"I": 100})
    assert_array_equal(model.initial_population, np.array([0, 100, 0]))
    model.set_initial_population({"R": 1, "S": 50, "I": 99})
    assert_array_equal(model.initial_population, np.array([50, 99, 1]))
