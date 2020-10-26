"""
Test that mixing matrices are built correctly.
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer.model import StratifiedModel
from summer.constants import BirthApproach

MODEL_KWARGS = {
    "times": np.array([0.0, 1, 2, 3, 4, 5]),
    "compartment_names": ["S", "I", "R"],
    "initial_conditions": {"I": 10},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 1000,
    "infectious_compartments": ["I"],
    "birth_approach": BirthApproach.NO_BIRTH,
    "entry_compartment": "S",
}


def test_no_mixing_matrix():
    """
    Test that we are using the default 'null-op' mixing matrix when
    we have a no user-specified mixing matrix
    """
    model = StratifiedModel(**MODEL_KWARGS)
    default_matrix = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=default_matrix,
    )

    # We should get the default mixing matrix
    actual_mixing = model.get_mixing_matrix(0)
    assert_array_equal(actual_mixing, default_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model.get_mixing_matrix(123)
    assert_array_equal(actual_mixing, default_matrix)

    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=None,
    )

    # We should get the default mixing matrix
    actual_mixing = model.get_mixing_matrix(0)
    assert_array_equal(actual_mixing, default_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model.get_mixing_matrix(123)
    assert_array_equal(actual_mixing, default_matrix)


def test_single_static_mixing_matrix():
    """
    Test that we are using the correct mixing matrix when
    we have a single static mixing matrix
    """
    model = StratifiedModel(**MODEL_KWARGS)
    agegroup_matrix = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=agegroup_matrix,
    )
    # We should get the age mixing matrix
    actual_mixing = model.get_mixing_matrix(0)
    assert_array_equal(actual_mixing, agegroup_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model.get_mixing_matrix(123)
    assert_array_equal(actual_mixing, agegroup_matrix)


def test_single_dynamic_mixing_matrix():
    """
    Test that we are using the correct mixing matrix when
    we have a single dynamic mixing matrix
    """
    model = StratifiedModel(**MODEL_KWARGS)
    agegroup_matrix = lambda t: t * np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=agegroup_matrix,
    )
    # We should get the age mixing matrix
    actual_mixing = model.get_mixing_matrix(1)
    assert_array_equal(actual_mixing, agegroup_matrix(1))
    # Dynamic matrices should change over time
    actual_mixing = model.get_mixing_matrix(123)
    assert_array_equal(actual_mixing, agegroup_matrix(123))


def test_multiple_static_mixing_matrix():
    """
    Test that we are using the correct mixing matrix when
    we have multiple static mixing matrices
    """
    model = StratifiedModel(**MODEL_KWARGS)
    agegroup_matrix = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=agegroup_matrix,
    )
    location_matrix = np.array([[11, 13], [17, 19]])
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=location_matrix,
    )
    expected_mixing = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    # We should get the Kronecker product of the two matrices
    actual_mixing = model.get_mixing_matrix(0)
    assert_array_equal(actual_mixing, expected_mixing)
    # Static matrices shouldn't change over time
    actual_mixing = model.get_mixing_matrix(123)
    assert_array_equal(actual_mixing, expected_mixing)
    # Double check that we calculated the Kronecker product correctly
    kron_mixing = np.kron(agegroup_matrix, location_matrix)
    assert_array_equal(expected_mixing, kron_mixing)


def test_multiple_dynamic_mixing_matrices():
    """
    Test that we are using the correct mixing matrix when
    we have multiple dynamic mixing matrices
    """
    model = StratifiedModel(**MODEL_KWARGS)
    agegroup_matrix = lambda t: t * np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=agegroup_matrix,
    )
    location_matrix = lambda t: t * np.array([[11, 13], [17, 19]])
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        mixing_matrix=location_matrix,
    )
    expected_mixing = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    # We should get the Kronecker product of the two matrices
    actual_mixing = model.get_mixing_matrix(1)
    assert_array_equal(actual_mixing, expected_mixing)
    # Double check that we calculated the Kronecker product correctly
    kron_mixing = np.kron(agegroup_matrix(1), location_matrix(1))
    assert_array_equal(expected_mixing, kron_mixing)
    # Dynamic matrices should change over time
    actual_mixing = model.get_mixing_matrix(5)
    assert_array_equal(actual_mixing, 25 * expected_mixing)
