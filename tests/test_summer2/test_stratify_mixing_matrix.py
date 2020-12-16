"""
Test applying a stratification with a mixing matrix via stratify_with 
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from summer2 import Stratification, CompartmentalModel


def test_add_mixing_matrix_fails():
    """
    Ensure validation works when trying to add a mixing matrix.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "R"])
    mixing_matrix = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(mixing_matrix)
    # Expect this to fail because it's not a full stratification (no I compartment).
    with pytest.raises(AssertionError):
        model.stratify_with(strat)


def test_no_mixing_matrix():
    """
    Test that we are using the default 'null-op' mixing matrix when
    we have a no user-specified mixing matrix
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    model.stratify_with(strat)

    # We should get the default mixing matrix
    default_matrix = np.array([[1]])
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, default_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, default_matrix)
    # No mixing categories have been added.
    assert model._mixing_categories == [{}]


def test_no_mixing_matrix__with_previous_strat():
    """
    Test that we are using the default 'null-op' mixing matrix when
    we have a no user-specified mixing matrix and a stratification has already been applied
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    # Apply first stratification with a mixing matrix.
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    first_strat_matrix = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(first_strat_matrix)
    model.stratify_with(strat)

    # We should get the default mixing matrix
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, first_strat_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, first_strat_matrix)
    # Agegroup mixing categories have been added.
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]

    # Apply second stratification with no mixing matrix.
    strat = Stratification(name="location", strata=["work", "home"], compartments=["S", "I", "R"])
    model.stratify_with(strat)

    # We should get the same results as before.
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, first_strat_matrix)
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, first_strat_matrix)
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]


def test_single_static_mixing_matrix():
    """
    Test that we are using the correct mixing matrix when
    we have a single static mixing matrix
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    # Apply first stratification with a mixing matrix.
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    mixing_matrix = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(mixing_matrix)
    model.stratify_with(strat)

    # We should get the default mixing matrix
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, mixing_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, mixing_matrix)
    # Agegroup mixing categories have been added.
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]


def test_single_dynamic_mixing_matrix():
    """
    Test that we are using the correct mixing matrix when
    we have a single dynamic mixing matrix
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    # Apply a stratification with a dynamic mixing matrix.
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    dynamic_mixing_matrix = lambda t: t * np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(dynamic_mixing_matrix)
    model.stratify_with(strat)

    # We should get the dynamic mixing matrix
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, 0 * np.array([[2, 3], [5, 7]]))
    # Dynamic matrices should change over time
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, 123 * np.array([[2, 3], [5, 7]]))
    # Agegroup mixing categories have been added.
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]


def test_multiple_static_mixing_matrices():
    """
    Test that we are using the correct mixing matrix when
    we have multiple static mixing matrices
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    # Apply agegroup stratification with a static mixing matrix.
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    agegroup_mixing_matrix = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(agegroup_mixing_matrix)
    model.stratify_with(strat)
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]

    # Apply location stratification with a static mixing matrix.
    strat = Stratification(name="location", strata=["work", "home"], compartments=["S", "I", "R"])
    location_mixing_matrix = np.array([[11, 13], [17, 19]])
    strat.set_mixing_matrix(location_mixing_matrix)
    model.stratify_with(strat)
    assert model._mixing_categories == [
        {"agegroup": "child", "location": "work"},
        {"agegroup": "child", "location": "home"},
        {"agegroup": "adult", "location": "work"},
        {"agegroup": "adult", "location": "home"},
    ]

    # We expect the two 2x2 mixing matrices to be combined into a single big 4x4,
    # using the Kronecker product of the two.
    expected_mixing_matrix = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    # We should get the Kronecker product of the two matrices
    actual_mixing = model._get_mixing_matrix(0)
    assert_array_equal(actual_mixing, expected_mixing_matrix)
    # Static matrices shouldn't change over time
    actual_mixing = model._get_mixing_matrix(123)
    assert_array_equal(actual_mixing, expected_mixing_matrix)
    # Double check that we calculated the Kronecker product correctly
    kron_mixing = np.kron(agegroup_mixing_matrix, location_mixing_matrix)
    assert_array_equal(expected_mixing_matrix, kron_mixing)


def test_multiple_dynamic_mixing_matrices():
    """
    Test that we are using the correct mixing matrix when
    we have multiple dynamic mixing matrices
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    # Apply agegroup stratification with a static mixing matrix.
    strat = Stratification(name="agegroup", strata=["child", "adult"], compartments=["S", "I", "R"])
    agegroup_mixing_matrix = lambda t: t * np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(agegroup_mixing_matrix)
    model.stratify_with(strat)
    assert model._mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]

    # Apply location stratification with a static mixing matrix.
    strat = Stratification(name="location", strata=["work", "home"], compartments=["S", "I", "R"])
    location_mixing_matrix = lambda t: t * np.array([[11, 13], [17, 19]])
    strat.set_mixing_matrix(location_mixing_matrix)
    model.stratify_with(strat)
    assert model._mixing_categories == [
        {"agegroup": "child", "location": "work"},
        {"agegroup": "child", "location": "home"},
        {"agegroup": "adult", "location": "work"},
        {"agegroup": "adult", "location": "home"},
    ]

    # We expect the two 2x2 mixing matrices to be combined into a single big 4x4,
    # using the Kronecker product of the two.
    expected_mixing_matrix = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    # We should get the Kronecker product of the two matrices
    actual_mixing = model._get_mixing_matrix(1)
    assert_array_equal(actual_mixing, expected_mixing_matrix)
    # Double check that we calculated the Kronecker product correctly
    kron_mixing = np.kron(agegroup_mixing_matrix(1), location_mixing_matrix(1))
    assert_array_equal(expected_mixing_matrix, kron_mixing)
    # Dynamic matrices should change over time
    actual_mixing = model._get_mixing_matrix(5)
    assert_array_equal(actual_mixing, 25 * expected_mixing_matrix)
