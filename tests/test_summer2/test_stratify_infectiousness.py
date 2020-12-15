"""
Ensure infectiousness adjustments are applied correctly in stratification.
See See https://parasiteecology.wordpress.com/2013/10/17/density-dependent-vs-frequency-dependent-disease-transmission/
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer2 import (
    CompartmentalModel,
    Stratification,
    StrainStratification,
    Compartment as C,
    adjust,
)


def test_strat_infectiousness__with_adjustments():
    """
    Ensure multiply infectiousness adjustment is applied.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 900, "I": 100})
    strat = Stratification("age", ["baby", "child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"baby": 0.1, "child": 0.3, "adult": 0.6})
    strat.add_infectiousness_adjustments(
        "I", {"child": adjust.Multiply(3), "adult": adjust.Multiply(0.5), "baby": None}
    )
    model.stratify_with(strat)
    assert_array_equal(
        model.initial_population,
        np.array([90, 270, 540, 10, 30, 60, 0, 0, 0]),
    )

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"],
        np.array([0, 0, 0, 1, 3, 0.5, 0, 0, 0]),
    )

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)

    # Get multipliers
    infectees = model.compartments[0:3]
    infectors = model.compartments[3:6]

    expected_density = 10 * 1 + 30 * 3 + 60 * 0.5
    expected_frequency = expected_density / 1000
    for infectee, infector in zip(infectees, infectors):
        assert model._get_infection_density_multiplier(infectee, infector) == expected_density

    for infectee, infector in zip(infectees, infectors):
        assert model._get_infection_frequency_multiplier(infectee, infector) == expected_frequency

    # Stratify again, now with overwrites
    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    strat.add_infectiousness_adjustments(
        "I", {"urban": adjust.Overwrite(1), "rural": adjust.Multiply(7)}
    )
    model.stratify_with(strat)
    assert_array_equal(
        model.initial_population,
        np.array([45, 45, 135, 135, 270.0, 270, 5, 5, 15, 15, 30, 30, 0, 0, 0, 0, 0, 0]),
    )

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"],
        np.array([0, 0, 0, 0, 0, 0, 1, 7, 1, 21, 1, 3.5, 0, 0, 0, 0, 0, 0]),
    )
    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)

    # Get multipliers
    infectees = model.compartments[0:6]
    infectors = model.compartments[6:12]
    expected_density = 5 * 1 + 5 * 7 + 15 * 1 + 15 * 21 + 30 * 1 + 30 * 3.5
    expected_frequency = expected_density / 1000
    for infectee, infector in zip(infectees, infectors):
        assert model._get_infection_density_multiplier(infectee, infector) == expected_density

    for infectee, infector in zip(infectees, infectors):
        assert model._get_infection_frequency_multiplier(infectee, infector) == expected_frequency
