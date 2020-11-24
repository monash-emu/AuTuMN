"""
Ensure that the CompartmentalModel model produces the correct force of infection multipliers
See https://parasiteecology.wordpress.com/2013/10/17/density-dependent-vs-frequency-dependent-disease-transmission/
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer2.model import CompartmentalModel, Stratification, Compartment as C


def test_basic_get_infection_multiplier():
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    model._prepare_to_run()
    model._prepare_time_step(0, model.initial_population)
    c_src = model.compartments[0]
    c_dst = model.compartments[1]
    multiplier = model._get_infection_frequency_multiplier(c_src, c_dst)
    assert multiplier == 10 / 1000
    multiplier = model._get_infection_density_multiplier(c_src, c_dst)
    assert multiplier == 10


def test_strat_get_infection_multiplier__with_age_strat_and_no_mixing():
    """
    Check FoI when a simple 2-strata stratification applied and no mixing matrix.
    Expect the same results as with the basic case.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    model.stratify_with(strat)
    assert model._mixing_categories == [{}]

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"], np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert_array_equal(model._category_matrix, np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
    assert model._category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[1000]]))
    assert_array_equal(model._infection_density["default"], np.array([[10.0]]))
    assert_array_equal(model._infection_frequency["default"], np.array([[0.01]]))

    # Get multipliers
    s_child = model.compartments[0]
    s_adult = model.compartments[1]
    i_child = model.compartments[2]
    i_adult = model.compartments[3]

    assert model._get_infection_density_multiplier(s_child, i_child) == 10.0
    assert model._get_infection_density_multiplier(s_adult, i_adult) == 10.0
    assert model._get_infection_frequency_multiplier(s_child, i_child) == 0.01
    assert model._get_infection_frequency_multiplier(s_adult, i_adult) == 0.01
    # Santiy check frequency-dependent force of infection
    assert 1000.0 * 0.01 == 10


def test_strat_get_infection_multiplier__with_age_strat_and_simple_mixing():
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Expect same frequency as before, different density.
    Note that the mixing matrix has different meanings for density / vs frequency.

    N.B Mixing matrix format.
    Columns are  the people who are infectors
    Rows are the people who are infected
    So matrix has following values

                  child               adult

      child       child -> child      adult -> child
      adult       child -> adult      adult -> adult

    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    strat.set_mixing_matrix(mixing_matrix)
    model.stratify_with(strat)

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(model.initial_population, np.array([495, 495, 5, 5, 0.0, 0.0]))

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"], np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert_array_equal(
        model._category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model._category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    child_density = 5
    adult_density = 5
    assert child_density == 0.5 * 5 + 0.5 * 5
    assert adult_density == 0.5 * 5 + 0.5 * 5
    assert_array_equal(
        model._infection_density["default"], np.array([[child_density], [adult_density]])
    )
    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == child_density / 500
    assert adult_freq == adult_density / 500
    assert_array_equal(
        model._infection_frequency["default"], np.array([[child_freq], [adult_freq]])
    )

    # Get multipliers
    s_child = model.compartments[0]
    s_adult = model.compartments[1]
    i_child = model.compartments[2]
    i_adult = model.compartments[3]
    assert model._get_infection_density_multiplier(s_child, i_child) == child_density
    assert model._get_infection_density_multiplier(s_adult, i_adult) == adult_density
    assert model._get_infection_frequency_multiplier(s_child, i_child) == child_freq
    assert model._get_infection_frequency_multiplier(s_adult, i_adult) == adult_freq
    # Santiy check frequency-dependent force of infection
    assert 500.0 * child_freq + 500.0 * adult_freq == 10


def test_strat_get_infection_multiplier__with_age_split_and_simple_mixing():
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Unequally split the children and adults.
    Expect same density as before, different frequency.
    Note that the mixing matrix has different meanings for density / vs frequency.
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    strat.set_mixing_matrix(mixing_matrix)
    strat.set_population_split({"child": 0.2, "adult": 0.8})
    model.stratify_with(strat)

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(model.initial_population, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"], np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert_array_equal(
        model._category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model._category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    child_density = 5
    adult_density = 5
    assert child_density == 0.5 * 5 + 0.5 * 5
    assert adult_density == 0.5 * 5 + 0.5 * 5
    assert_array_equal(
        model._infection_density["default"], np.array([[child_density], [adult_density]])
    )
    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert adult_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert_array_equal(
        model._infection_frequency["default"], np.array([[child_freq], [adult_freq]])
    )

    # Get multipliers
    s_child = model.compartments[0]
    s_adult = model.compartments[1]
    i_child = model.compartments[2]
    i_adult = model.compartments[3]

    assert model._get_infection_density_multiplier(s_child, i_child) == child_density
    assert model._get_infection_density_multiplier(s_adult, i_adult) == adult_density
    assert model._get_infection_frequency_multiplier(s_child, i_child) == child_freq
    assert model._get_infection_frequency_multiplier(s_adult, i_adult) == adult_freq
    # Santiy check frequency-dependent force of infection
    assert 200.0 * child_freq + 800.0 * adult_freq == 10


def test_strat_get_infection_multiplier__with_age_strat_and_mixing():
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Use a non-uniform mixing matrix
    """
    # Create a model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    mixing_matrix = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(mixing_matrix)
    strat.set_population_split({"child": 0.2, "adult": 0.8})
    model.stratify_with(strat)

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child"}),
        C("S", {"age": "adult"}),
        C("I", {"age": "child"}),
        C("I", {"age": "adult"}),
        C("R", {"age": "child"}),
        C("R", {"age": "adult"}),
    ]
    assert_array_equal(model.initial_population, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    assert_array_equal(
        model._compartment_infectiousness["default"], np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert_array_equal(
        model._category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model._category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[200.0], [800.0]]))
    child_density = 28
    adult_density = 66
    assert child_density == 2 * 2.0 + 3 * 8.0
    assert adult_density == 5 * 2.0 + 7 * 8.0
    assert_array_equal(
        model._infection_density["default"], np.array([[child_density], [adult_density]])
    )
    child_freq = 0.05
    adult_freq = 0.12000000000000001
    assert child_freq == 2 * 2.0 / 200 + 3 * 8.0 / 800
    assert adult_freq == 5 * 2.0 / 200 + 7 * 8.0 / 800

    assert_array_equal(
        model._infection_frequency["default"], np.array([[child_freq], [adult_freq]])
    )

    # Get multipliers
    s_child = model.compartments[0]
    s_adult = model.compartments[1]
    i_child = model.compartments[2]
    i_adult = model.compartments[3]
    assert model._get_infection_density_multiplier(s_child, i_child) == child_density
    assert model._get_infection_density_multiplier(s_adult, i_adult) == adult_density
    assert model._get_infection_frequency_multiplier(s_child, i_child) == child_freq
    assert model._get_infection_frequency_multiplier(s_adult, i_adult) == adult_freq


def test_strat_get_infection_multiplier__with_double_strat_and_no_mixing():
    """
    Check FoI when a two 2-strata stratificationz applied and no mixing matrix.
    Expect the same results as with the basic case.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    model.stratify_with(strat)
    strat = Stratification("location", ["work", "home"], ["S", "I", "R"])
    model.stratify_with(strat)

    assert model._mixing_categories == [{}]
    assert model.compartments == [
        C("S", {"age": "child", "location": "work"}),
        C("S", {"age": "child", "location": "home"}),
        C("S", {"age": "adult", "location": "work"}),
        C("S", {"age": "adult", "location": "home"}),
        C("I", {"age": "child", "location": "work"}),
        C("I", {"age": "child", "location": "home"}),
        C("I", {"age": "adult", "location": "work"}),
        C("I", {"age": "adult", "location": "home"}),
        C("R", {"age": "child", "location": "work"}),
        C("R", {"age": "child", "location": "home"}),
        C("R", {"age": "adult", "location": "work"}),
        C("R", {"age": "adult", "location": "home"}),
    ]
    expected_comp_vals = np.array([247.5, 247.5, 247.5, 247.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 0, 0])
    assert_array_equal(model.initial_population, expected_comp_vals)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    expected_compartment_infectiousness = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    expected_category_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    expected_lookup = {n: 0 for n in range(12)}
    assert_array_equal(
        model._compartment_infectiousness["default"], expected_compartment_infectiousness
    )
    assert_array_equal(model._category_matrix, expected_category_matrix)
    assert model._category_lookup == expected_lookup

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[1000]]))
    assert_array_equal(model._infection_density["default"], np.array([[10.0]]))
    assert_array_equal(model._infection_frequency["default"], np.array([[0.01]]))
    s_child_work = model.compartments[0]
    s_child_home = model.compartments[1]
    s_adult_work = model.compartments[2]
    s_adult_home = model.compartments[3]
    i_child_work = model.compartments[4]
    i_child_home = model.compartments[5]
    i_adult_work = model.compartments[6]
    i_adult_home = model.compartments[7]

    assert model._get_infection_density_multiplier(s_child_work, i_child_work) == 10.0
    assert model._get_infection_density_multiplier(s_child_home, i_child_home) == 10.0
    assert model._get_infection_density_multiplier(s_adult_work, i_adult_work) == 10.0
    assert model._get_infection_density_multiplier(s_adult_home, i_adult_home) == 10.0
    assert model._get_infection_frequency_multiplier(s_child_work, i_child_work) == 0.01
    assert model._get_infection_frequency_multiplier(s_child_home, i_child_home) == 0.01
    assert model._get_infection_frequency_multiplier(s_adult_work, i_adult_work) == 0.01
    assert model._get_infection_frequency_multiplier(s_adult_home, i_adult_home) == 0.01


def test_strat_get_infection_multiplier__with_double_strat_and_first_strat_mixing():
    """
    Check FoI when a two 2-strata stratification applied and the first stratification has a mixing matrix.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"child": 0.3, "adult": 0.7})
    age_mixing = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(age_mixing)
    model.stratify_with(strat)
    strat = Stratification("location", ["work", "home"], ["S", "I", "R"])
    model.stratify_with(strat)

    assert model._mixing_categories == [{"age": "child"}, {"age": "adult"}]
    assert model.compartments == [
        C("S", {"age": "child", "location": "work"}),
        C("S", {"age": "child", "location": "home"}),
        C("S", {"age": "adult", "location": "work"}),
        C("S", {"age": "adult", "location": "home"}),
        C("I", {"age": "child", "location": "work"}),
        C("I", {"age": "child", "location": "home"}),
        C("I", {"age": "adult", "location": "work"}),
        C("I", {"age": "adult", "location": "home"}),
        C("R", {"age": "child", "location": "work"}),
        C("R", {"age": "child", "location": "home"}),
        C("R", {"age": "adult", "location": "work"}),
        C("R", {"age": "adult", "location": "home"}),
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.initial_population, expected_comp_vals)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    exp_lookup = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 1, 11: 1}
    exp_matrix = np.array(
        [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]
    )
    exp_mults = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert_array_equal(model._compartment_infectiousness["default"], exp_mults)
    assert_array_equal(model._category_matrix, exp_matrix)
    assert model._category_lookup == exp_lookup

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[300.0], [700.0]]))
    child_density = 27
    adult_density = 64
    assert child_density == 2 * 3 + 3 * 7
    assert adult_density == 5 * 3 + 7 * 7
    assert_array_equal(
        model._infection_density["default"], np.array([[child_density], [adult_density]])
    )
    child_freq = 2 * 3 / 300 + 3 * 7 / 700
    adult_freq = 5 * 3 / 300 + 7 * 7 / 700
    assert_array_equal(
        model._infection_frequency["default"], np.array([[child_freq], [adult_freq]])
    )

    # Get multipliers
    s_child_work = model.compartments[0]
    s_child_home = model.compartments[1]
    s_adult_work = model.compartments[2]
    s_adult_home = model.compartments[3]
    i_child_work = model.compartments[4]
    i_child_home = model.compartments[5]
    i_adult_work = model.compartments[6]
    i_adult_home = model.compartments[7]

    assert model._get_infection_density_multiplier(s_child_work, i_child_work) == child_density
    assert model._get_infection_density_multiplier(s_child_home, i_child_home) == child_density
    assert model._get_infection_density_multiplier(s_adult_work, i_adult_work) == adult_density
    assert model._get_infection_density_multiplier(s_adult_home, i_adult_home) == adult_density
    assert model._get_infection_frequency_multiplier(s_child_work, i_child_work) == child_freq
    assert model._get_infection_frequency_multiplier(s_child_home, i_child_home) == child_freq
    assert model._get_infection_frequency_multiplier(s_adult_work, i_adult_work) == adult_freq
    assert model._get_infection_frequency_multiplier(s_adult_home, i_adult_home) == adult_freq


def test_strat_get_infection_multiplier__with_double_strat_and_second_strat_mixing():
    """
    Check FoI when a two 2-strata stratification applied and the second stratification has a mixing matrix.
    """
    # Create the model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"child": 0.3, "adult": 0.7})
    model.stratify_with(strat)
    strat = Stratification("location", ["work", "home"], ["S", "I", "R"])
    location_mixing = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(location_mixing)
    model.stratify_with(strat)

    assert model._mixing_categories == [{"location": "work"}, {"location": "home"}]
    assert model.compartments == [
        C("S", {"age": "child", "location": "work"}),
        C("S", {"age": "child", "location": "home"}),
        C("S", {"age": "adult", "location": "work"}),
        C("S", {"age": "adult", "location": "home"}),
        C("I", {"age": "child", "location": "work"}),
        C("I", {"age": "child", "location": "home"}),
        C("I", {"age": "adult", "location": "work"}),
        C("I", {"age": "adult", "location": "home"}),
        C("R", {"age": "child", "location": "work"}),
        C("R", {"age": "child", "location": "home"}),
        C("R", {"age": "adult", "location": "work"}),
        C("R", {"age": "adult", "location": "home"}),
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.initial_population, expected_comp_vals)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    exp_lookup = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1}
    exp_matrix = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    exp_mults = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert_array_equal(model._compartment_infectiousness["default"], exp_mults)
    assert_array_equal(model._category_matrix, exp_matrix)
    assert model._category_lookup == exp_lookup

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    assert_array_equal(model._category_populations, np.array([[500.0], [500.0]]))
    work_density = 25
    home_density = 60
    assert work_density == 2 * 5 + 3 * 5
    assert home_density == 5 * 5 + 7 * 5
    assert_array_equal(
        model._infection_density["default"], np.array([[work_density], [home_density]])
    )
    work_freq = 2 * 5 / 500 + 3 * 5 / 500
    home_freq = 5 * 5 / 500 + 7 * 5 / 500
    assert_array_equal(model._infection_frequency["default"], np.array([[work_freq], [home_freq]]))

    # Get multipliers
    s_child_work = model.compartments[0]
    s_child_home = model.compartments[1]
    s_adult_work = model.compartments[2]
    s_adult_home = model.compartments[3]
    i_child_work = model.compartments[4]
    i_child_home = model.compartments[5]
    i_adult_work = model.compartments[6]
    i_adult_home = model.compartments[7]

    assert model._get_infection_density_multiplier(s_child_work, i_child_work) == work_density
    assert model._get_infection_density_multiplier(s_child_home, i_child_home) == home_density
    assert model._get_infection_density_multiplier(s_adult_work, i_adult_work) == work_density
    assert model._get_infection_density_multiplier(s_adult_home, i_adult_home) == home_density
    assert model._get_infection_frequency_multiplier(s_child_work, i_child_work) == work_freq
    assert model._get_infection_frequency_multiplier(s_child_home, i_child_home) == home_freq
    assert model._get_infection_frequency_multiplier(s_adult_work, i_adult_work) == work_freq
    assert model._get_infection_frequency_multiplier(s_adult_home, i_adult_home) == home_freq


def test_strat_get_infection_multiplier__with_double_strat_and_both_strats_mixing():
    """
    Check FoI when a two 2-strata stratification applied and both stratifications have a mixing matrix.
    """
    # Create the model
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.set_initial_population(distribution={"S": 990, "I": 10})
    strat = Stratification("age", ["child", "adult"], ["S", "I", "R"])
    strat.set_population_split({"child": 0.3, "adult": 0.7})
    am = np.array([[2, 3], [5, 7]])
    strat.set_mixing_matrix(am)
    model.stratify_with(strat)
    strat = Stratification("location", ["work", "home"], ["S", "I", "R"])
    lm = np.array([[11, 13], [17, 19]])
    strat.set_mixing_matrix(lm)
    model.stratify_with(strat)

    expected_mixing = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    assert_array_equal(model._get_mixing_matrix(0), expected_mixing)
    assert model._mixing_categories == [
        {"age": "child", "location": "work"},
        {"age": "child", "location": "home"},
        {"age": "adult", "location": "work"},
        {"age": "adult", "location": "home"},
    ]
    assert model.compartments == [
        C("S", {"age": "child", "location": "work"}),
        C("S", {"age": "child", "location": "home"}),
        C("S", {"age": "adult", "location": "work"}),
        C("S", {"age": "adult", "location": "home"}),
        C("I", {"age": "child", "location": "work"}),
        C("I", {"age": "child", "location": "home"}),
        C("I", {"age": "adult", "location": "work"}),
        C("I", {"age": "adult", "location": "home"}),
        C("R", {"age": "child", "location": "work"}),
        C("R", {"age": "child", "location": "home"}),
        C("R", {"age": "adult", "location": "work"}),
        C("R", {"age": "adult", "location": "home"}),
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.initial_population, expected_comp_vals)

    # Do pre-run force of infection calcs.
    model._prepare_to_run()
    exp_lookup = {
        0: 0,  # S age: child location: work -> children at work
        1: 1,  # S age: child location: home -> children at home
        2: 2,  # S age: adult location: work -> adults at work
        3: 3,  # S age: adult location: home -> adults at home
        4: 0,  # I age: child location: work -> children at work
        5: 1,  # I age: child location: home -> children at home
        6: 2,  # I age: adult location: work -> adults at work
        7: 3,  # I age: adult location: home -> adults at home
        8: 0,  # R age: child location: work -> children at work
        9: 1,  # R age: child location: home -> children at home
        10: 2,  # R age: adult location: work -> adults at work
        11: 3,  # R age: adult location: home -> adults at home
    }
    exp_matrix = np.array(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # children at work
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # children at home
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # adults at work
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # adults at home
        ]
    )
    exp_mults = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert_array_equal(model._compartment_infectiousness["default"], exp_mults)
    assert_array_equal(model._category_matrix, exp_matrix)
    assert model._category_lookup == exp_lookup

    # Do pre-iteration force of infection calcs
    model._prepare_time_step(0, model.initial_population)
    exp_pops = np.array(
        [
            [150],  # children at work
            [150],  # children at home
            [350],  # adults at work
            [350],  # adults at home
        ]
    )
    assert_array_equal(model._category_populations, exp_pops)
    child_work_density = 2 * 11 * 1.5 + 2 * 13 * 1.5 + 3 * 11 * 3.5 + 3 * 13 * 3.5
    child_home_density = 2 * 17 * 1.5 + 2 * 19 * 1.5 + 3 * 17 * 3.5 + 3 * 19 * 3.5
    adult_work_density = 5 * 11 * 1.5 + 5 * 13 * 1.5 + 7 * 11 * 3.5 + 7 * 13 * 3.5
    adult_home_density = 5 * 17 * 1.5 + 5 * 19 * 1.5 + 7 * 17 * 3.5 + 7 * 19 * 3.5
    child_work_freq = (
        2 * 11 * 1.5 / 150 + 2 * 13 * 1.5 / 150 + 3 * 11 * 3.5 / 350 + 3 * 13 * 3.5 / 350
    )
    child_home_freq = (
        2 * 17 * 1.5 / 150 + 2 * 19 * 1.5 / 150 + 3 * 17 * 3.5 / 350 + 3 * 19 * 3.5 / 350
    )
    adult_work_freq = (
        5 * 11 * 1.5 / 150 + 5 * 13 * 1.5 / 150 + 7 * 11 * 3.5 / 350 + 7 * 13 * 3.5 / 350
    )
    adult_home_freq = (
        5 * 17 * 1.5 / 150 + 5 * 19 * 1.5 / 150 + 7 * 17 * 3.5 / 350 + 7 * 19 * 3.5 / 350
    )
    exp_density = np.array(
        [
            [child_work_density],  # children at work
            [child_home_density],  # children at home
            [adult_work_density],  # adults at work
            [adult_home_density],  # adults at home
        ]
    )
    exp_frequency = np.array(
        [
            [child_work_freq],  # children at work
            [child_home_freq],  # children at home
            [adult_work_freq],  # adults at work
            [adult_home_freq],  # adults at home
        ]
    )
    assert_array_equal(model._infection_density["default"], exp_density)
    assert_allclose(model._infection_frequency["default"], exp_frequency, rtol=0, atol=1e-9)

    # Get multipliers
    s_child_work = model.compartments[0]
    s_child_home = model.compartments[1]
    s_adult_work = model.compartments[2]
    s_adult_home = model.compartments[3]
    i_child_work = model.compartments[4]
    i_child_home = model.compartments[5]
    i_adult_work = model.compartments[6]
    i_adult_home = model.compartments[7]

    assert (
        abs(
            model._get_infection_density_multiplier(s_child_work, i_child_work) - child_work_density
        )
        <= 1e-9
    )
    assert (
        abs(
            model._get_infection_density_multiplier(s_child_home, i_child_home) - child_home_density
        )
        <= 1e-9
    )
    assert (
        abs(
            model._get_infection_density_multiplier(s_adult_work, i_adult_work) - adult_work_density
        )
        <= 1e-9
    )
    assert (
        abs(
            model._get_infection_density_multiplier(s_adult_home, i_adult_home) - adult_home_density
        )
        <= 1e-9
    )
    assert (
        abs(model._get_infection_frequency_multiplier(s_child_work, i_child_work) - child_work_freq)
        <= 1e-9
    )
    assert (
        abs(model._get_infection_frequency_multiplier(s_child_home, i_child_home) - child_home_freq)
        <= 1e-9
    )
    assert (
        abs(model._get_infection_frequency_multiplier(s_adult_work, i_adult_work) - adult_work_freq)
        <= 1e-9
    )
    assert (
        abs(model._get_infection_frequency_multiplier(s_adult_home, i_adult_home) - adult_home_freq)
        <= 1e-9
    )
