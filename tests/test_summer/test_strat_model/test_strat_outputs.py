"""
End-to-end tests for the StratifiedModel - a disease agnostic compartmental model from SUMMER
"""
import numpy as np
from numpy.testing import assert_allclose

from summer.model import StratifiedModel
from summer.constants import (
    BirthApproach,
    IntegrationType,
)


def test_strat_model__with_age__expect_ageing():
    """
    Ensure that a module with age stratification produces ageing flows,
    and the correct output.
    """
    pop = 1000
    model = StratifiedModel(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=["S", "I"],
        initial_conditions={"S": pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
        infectious_compartments=["I"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="S",
    )
    # Add basic age stratification
    model.stratify("age", strata_request=[0, 5, 15, 60], compartments_to_stratify=["S", "I"])

    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to generally get older, but no one should die or get sick
    expected_arr = np.array(
        [
            [250.0, 250.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0],
            [204.8, 269.1, 270.3, 255.8, 0.0, 0.0, 0.0, 0.0],
            [167.7, 278.8, 291.5, 262.0, 0.0, 0.0, 0.0, 0.0],
            [137.3, 281.2, 312.8, 268.7, 0.0, 0.0, 0.0, 0.0],
            [112.5, 277.9, 333.7, 275.9, 0.0, 0.0, 0.0, 0.0],
            [92.1, 270.8, 353.5, 283.6, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


def test_strat_model__with_age_and_starting_proportion__expect_ageing():
    """
    Ensure that a module with age stratification and starting proporptions
    produces ageing flows, and the correct output.
    """
    pop = 1000
    model = StratifiedModel(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=["S", "I"],
        initial_conditions={"S": pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
        infectious_compartments=["I"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="S",
    )
    # Add basic age stratification
    model.stratify(
        "age",
        strata_request=[0, 5, 15, 60],
        compartments_to_stratify=["S", "I"],
        comp_split_props={"0": 0.8, "5": 0.1, "15": 0.1},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to generally get older, but no one should die or get sick.
    # Expect initial distribution of ages to be set according to "requested_proportions".
    expected_arr = np.array(
        [
            [800.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [655.0, 228.3, 114.4, 2.4, 0.0, 0.0, 0.0, 0.0],
            [536.3, 319.3, 139.3, 5.2, 0.0, 0.0, 0.0, 0.0],
            [439.1, 381.3, 171.1, 8.6, 0.0, 0.0, 0.0, 0.0],
            [359.5, 420.5, 207.2, 12.8, 0.0, 0.0, 0.0, 0.0],
            [294.4, 442.4, 245.4, 17.8, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


def test_strat_model__with_locations__expect_no_change():
    """
    Ensure that a module with location stratification populates locations correctly.
    """
    pop = 1000
    model = StratifiedModel(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=["S", "I"],
        initial_conditions={"S": pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
        infectious_compartments=["I"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="S",
    )
    # Add basic location stratification
    model.stratify(
        "location",
        strata_request=["rural", "urban", "prison"],
        compartments_to_stratify=["S", "I"],
        comp_split_props={"rural": 0.44, "urban": 0.55, "prison": 0.01},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to start in their locations, then nothing should change,
    expected_arr = np.array(
        [
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
            [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


def _get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return np.linspace(start_year, end_year, n_iter)
