"""
Ensure that the StratifiedModel model produces the correct infectious multipliers
See https://parasiteecology.wordpress.com/2013/10/17/density-dependent-vs-frequency-dependent-disease-transmission/
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer.model import StratifiedModel
from summer.constants import Flow, BirthApproach
from summer.compartment import Compartment

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


def test_strat_basic_get_infection_multipier():
    """
    Check basic case for force of infection (FoI).
    No stratifications / heterogeneous mixing.
    Expect the same results as with the EpiModel.
    """
    # Create a model
    model = StratifiedModel(**MODEL_KWARGS)
    assert model.mixing_categories == [{}]

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    assert_array_equal(model.infectiousness_multipliers, np.array([0.0, 1.0, 0.0]))
    assert_array_equal(model.category_matrix, np.array([[1.0, 1.0, 1.0]]))
    assert model.category_lookup == {0: 0, 1: 0, 2: 0}

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[1000]]))
    assert_array_equal(model.infection_density, np.array([[10.0]]))
    assert_array_equal(model.infection_frequency, np.array([[0.01]]))

    # Get multipliers
    susceptible = model.compartment_names[0]
    assert model.get_infection_density_multipier(susceptible) == 10.0
    assert model.get_infection_frequency_multipier(susceptible) == 0.01
    # Santiy check frequency-dependent force of infection
    assert 1000.0 * 0.01 == 10


def test_strat_get_infection_multipier__with_age_strat_and_no_mixing():
    """
    Check FoI when a simple 2-strata stratification applied and no mixing matrix.
    Expect the same results as with the basic case.
    """
    # Create a model
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    assert model.mixing_categories == [{}]

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    assert_array_equal(model.infectiousness_multipliers, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert_array_equal(model.category_matrix, np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
    assert model.category_lookup == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[1000]]))
    assert_array_equal(model.infection_density, np.array([[10.0]]))
    assert_array_equal(model.infection_frequency, np.array([[0.01]]))

    # Get multipliers
    s_child = model.compartment_names[0]
    s_adult = model.compartment_names[1]
    assert model.get_infection_density_multipier(s_child) == 10.0
    assert model.get_infection_density_multipier(s_adult) == 10.0
    assert model.get_infection_frequency_multipier(s_child) == 0.01
    assert model.get_infection_frequency_multipier(s_adult) == 0.01
    # Santiy check frequency-dependent force of infection
    assert 1000.0 * 0.01 == 10


def test_strat_get_infection_multipier__with_age_strat_and_simple_mixing():
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
    # Create a model
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=mixing_matrix,
    )
    assert model.mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]
    assert model.compartment_names == [
        "SXagegroup_child",
        "SXagegroup_adult",
        "IXagegroup_child",
        "IXagegroup_adult",
        "RXagegroup_child",
        "RXagegroup_adult",
    ]
    assert_array_equal(model.compartment_values, np.array([495, 495, 5, 5, 0.0, 0.0]))

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    assert_array_equal(model.infectiousness_multipliers, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert_array_equal(
        model.category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model.category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    child_density = 5
    adult_density = 5
    assert child_density == 0.5 * 5 + 0.5 * 5
    assert adult_density == 0.5 * 5 + 0.5 * 5
    assert_array_equal(model.infection_density, np.array([[child_density], [adult_density]]))
    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == child_density / 500
    assert adult_freq == adult_density / 500
    assert_array_equal(model.infection_frequency, np.array([[child_freq], [adult_freq]]))

    # Get multipliers
    s_child = model.compartment_names[0]
    s_adult = model.compartment_names[1]
    assert model.get_infection_density_multipier(s_child) == child_density
    assert model.get_infection_density_multipier(s_adult) == adult_density
    assert model.get_infection_frequency_multipier(s_child) == child_freq
    assert model.get_infection_frequency_multipier(s_adult) == adult_freq
    # Santiy check frequency-dependent force of infection
    assert 500.0 * child_freq + 500.0 * adult_freq == 10


def test_strat_get_infection_multipier__with_age_split_and_simple_mixing():
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Unequally split the children and adults.
    Expect same density as before, different frequency.
    Note that the mixing matrix has different meanings for density / vs frequency.
    """
    # Create a model
    mixing_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={"child": 0.2, "adult": 0.8,},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=mixing_matrix,
    )
    assert model.mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]
    assert model.compartment_names == [
        "SXagegroup_child",
        "SXagegroup_adult",
        "IXagegroup_child",
        "IXagegroup_adult",
        "RXagegroup_child",
        "RXagegroup_adult",
    ]
    assert_array_equal(model.compartment_values, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    assert_array_equal(model.infectiousness_multipliers, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert_array_equal(
        model.category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model.category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    child_density = 5
    adult_density = 5
    assert child_density == 0.5 * 5 + 0.5 * 5
    assert adult_density == 0.5 * 5 + 0.5 * 5
    assert_array_equal(model.infection_density, np.array([[child_density], [adult_density]]))
    child_freq = 0.01
    adult_freq = 0.01
    assert child_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert adult_freq == 0.5 * 2 / 200 + 0.5 * 8 / 800
    assert_array_equal(model.infection_frequency, np.array([[child_freq], [adult_freq]]))

    # Get multipliers
    s_child = model.compartment_names[0]
    s_adult = model.compartment_names[1]
    assert model.get_infection_density_multipier(s_child) == child_density
    assert model.get_infection_density_multipier(s_adult) == adult_density
    assert model.get_infection_frequency_multipier(s_child) == child_freq
    assert model.get_infection_frequency_multipier(s_adult) == adult_freq
    # Santiy check frequency-dependent force of infection
    assert 200.0 * child_freq + 800.0 * adult_freq == 10


def test_strat_get_infection_multipier__with_age_strat_and_mixing():
    """
    Check FoI when a simple 2-strata stratification applied AND heteregeneous mixing.
    Use a non-uniform mixing matrix
    """
    # Create a model
    mixing_matrix = np.array([[2, 3], [5, 7]])
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={"child": 0.2, "adult": 0.8,},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=mixing_matrix,
    )
    assert model.mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]
    assert model.compartment_names == [
        "SXagegroup_child",
        "SXagegroup_adult",
        "IXagegroup_child",
        "IXagegroup_adult",
        "RXagegroup_child",
        "RXagegroup_adult",
    ]
    assert_array_equal(model.compartment_values, np.array([198.0, 792.0, 2.0, 8.0, 0.0, 0.0]))

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    assert_array_equal(model.infectiousness_multipliers, np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert_array_equal(
        model.category_matrix,
        np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
    )
    assert model.category_lookup == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[200.0], [800.0]]))
    child_density = 28
    adult_density = 66
    assert child_density == 2 * 2.0 + 3 * 8.0
    assert adult_density == 5 * 2.0 + 7 * 8.0
    assert_array_equal(model.infection_density, np.array([[child_density], [adult_density]]))
    child_freq = 0.05
    adult_freq = 0.12000000000000001
    assert child_freq == 2 * 2.0 / 200 + 3 * 8.0 / 800
    assert adult_freq == 5 * 2.0 / 200 + 7 * 8.0 / 800

    assert_array_equal(model.infection_frequency, np.array([[child_freq], [adult_freq]]))

    # Get multipliers
    s_child = model.compartment_names[0]
    s_adult = model.compartment_names[1]
    assert model.get_infection_density_multipier(s_child) == child_density
    assert model.get_infection_density_multipier(s_adult) == adult_density
    assert model.get_infection_frequency_multipier(s_child) == child_freq
    assert model.get_infection_frequency_multipier(s_adult) == adult_freq


def test_strat_get_infection_multipier__with_double_strat_and_no_mixing():
    """
    Check FoI when a two 2-strata stratificationz applied and no mixing matrix.
    Expect the same results as with the basic case.
    """
    # Create a model
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    assert model.mixing_categories == [{}]
    assert model.compartment_names == [
        "SXagegroup_childXlocation_work",
        "SXagegroup_childXlocation_home",
        "SXagegroup_adultXlocation_work",
        "SXagegroup_adultXlocation_home",
        "IXagegroup_childXlocation_work",
        "IXagegroup_childXlocation_home",
        "IXagegroup_adultXlocation_work",
        "IXagegroup_adultXlocation_home",
        "RXagegroup_childXlocation_work",
        "RXagegroup_childXlocation_home",
        "RXagegroup_adultXlocation_work",
        "RXagegroup_adultXlocation_home",
    ]
    expected_comp_vals = np.array([247.5, 247.5, 247.5, 247.5, 2.5, 2.5, 2.5, 2.5, 0, 0, 0, 0])
    assert_array_equal(model.compartment_values, expected_comp_vals)

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    expected_infectiousness_multipliers = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    expected_category_matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    expected_lookup = {n: 0 for n in range(12)}
    assert_array_equal(model.infectiousness_multipliers, expected_infectiousness_multipliers)
    assert_array_equal(model.category_matrix, expected_category_matrix)
    assert model.category_lookup == expected_lookup

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[1000]]))
    assert_array_equal(model.infection_density, np.array([[10.0]]))
    assert_array_equal(model.infection_frequency, np.array([[0.01]]))
    s_child_work = model.compartment_names[0]
    s_child_home = model.compartment_names[1]
    s_adult_work = model.compartment_names[2]
    s_adult_home = model.compartment_names[3]
    assert model.get_infection_density_multipier(s_child_work) == 10.0
    assert model.get_infection_density_multipier(s_child_home) == 10.0
    assert model.get_infection_density_multipier(s_adult_work) == 10.0
    assert model.get_infection_density_multipier(s_adult_home) == 10.0
    assert model.get_infection_frequency_multipier(s_child_work) == 0.01
    assert model.get_infection_frequency_multipier(s_child_home) == 0.01
    assert model.get_infection_frequency_multipier(s_adult_work) == 0.01
    assert model.get_infection_frequency_multipier(s_adult_home) == 0.01


def test_strat_get_infection_multipier__with_double_strat_and_first_strat_mixing():
    """
    Check FoI when a two 2-strata stratification applied and the first stratification has a mixing matrix.
    """
    # Create the model
    model = StratifiedModel(**MODEL_KWARGS)
    age_mixing = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={"child": 0.3, "adult": 0.7,},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=age_mixing,
    )
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    assert model.mixing_categories == [{"agegroup": "child"}, {"agegroup": "adult"}]
    assert model.compartment_names == [
        "SXagegroup_childXlocation_work",
        "SXagegroup_childXlocation_home",
        "SXagegroup_adultXlocation_work",
        "SXagegroup_adultXlocation_home",
        "IXagegroup_childXlocation_work",
        "IXagegroup_childXlocation_home",
        "IXagegroup_adultXlocation_work",
        "IXagegroup_adultXlocation_home",
        "RXagegroup_childXlocation_work",
        "RXagegroup_childXlocation_home",
        "RXagegroup_adultXlocation_work",
        "RXagegroup_adultXlocation_home",
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.compartment_values, expected_comp_vals)

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    exp_lookup = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 1, 11: 1}
    exp_matrix = np.array(
        [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]
    )
    exp_mults = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert_array_equal(model.infectiousness_multipliers, exp_mults)
    assert_array_equal(model.category_matrix, exp_matrix)
    assert model.category_lookup == exp_lookup

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[300.0], [700.0]]))
    child_density = 27
    adult_density = 64
    assert child_density == 2 * 3 + 3 * 7
    assert adult_density == 5 * 3 + 7 * 7
    assert_array_equal(model.infection_density, np.array([[child_density], [adult_density]]))
    child_freq = 2 * 3 / 300 + 3 * 7 / 700
    adult_freq = 5 * 3 / 300 + 7 * 7 / 700
    assert_array_equal(model.infection_frequency, np.array([[child_freq], [adult_freq]]))

    # Get multipliers
    s_child_work = model.compartment_names[0]
    s_child_home = model.compartment_names[1]
    s_adult_work = model.compartment_names[2]
    s_adult_home = model.compartment_names[3]
    assert model.get_infection_density_multipier(s_child_work) == child_density
    assert model.get_infection_density_multipier(s_child_home) == child_density
    assert model.get_infection_density_multipier(s_adult_work) == adult_density
    assert model.get_infection_density_multipier(s_adult_home) == adult_density
    assert model.get_infection_frequency_multipier(s_child_work) == child_freq
    assert model.get_infection_frequency_multipier(s_child_home) == child_freq
    assert model.get_infection_frequency_multipier(s_adult_work) == adult_freq
    assert model.get_infection_frequency_multipier(s_adult_home) == adult_freq


def test_strat_get_infection_multipier__with_double_strat_and_second_strat_mixing():
    """
    Check FoI when a two 2-strata stratification applied and the second stratification has a mixing matrix.
    """
    # Create the model
    model = StratifiedModel(**MODEL_KWARGS)
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={"child": 0.3, "adult": 0.7,},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=None,
    )
    location_mixing = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=location_mixing,
    )
    assert model.mixing_categories == [{"location": "work"}, {"location": "home"}]
    assert model.compartment_names == [
        "SXagegroup_childXlocation_work",
        "SXagegroup_childXlocation_home",
        "SXagegroup_adultXlocation_work",
        "SXagegroup_adultXlocation_home",
        "IXagegroup_childXlocation_work",
        "IXagegroup_childXlocation_home",
        "IXagegroup_adultXlocation_work",
        "IXagegroup_adultXlocation_home",
        "RXagegroup_childXlocation_work",
        "RXagegroup_childXlocation_home",
        "RXagegroup_adultXlocation_work",
        "RXagegroup_adultXlocation_home",
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.compartment_values, expected_comp_vals)

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    exp_lookup = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1}
    exp_matrix = np.array(
        [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],]
    )
    exp_mults = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    assert_array_equal(model.infectiousness_multipliers, exp_mults)
    assert_array_equal(model.category_matrix, exp_matrix)
    assert model.category_lookup == exp_lookup

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    assert_array_equal(model.category_populations, np.array([[500.0], [500.0]]))
    work_density = 25
    home_density = 60
    assert work_density == 2 * 5 + 3 * 5
    assert home_density == 5 * 5 + 7 * 5
    assert_array_equal(model.infection_density, np.array([[work_density], [home_density]]))
    work_freq = 2 * 5 / 500 + 3 * 5 / 500
    home_freq = 5 * 5 / 500 + 7 * 5 / 500
    assert_array_equal(model.infection_frequency, np.array([[work_freq], [home_freq]]))

    # Get multipliers
    s_child_work = model.compartment_names[0]
    s_child_home = model.compartment_names[1]
    s_adult_work = model.compartment_names[2]
    s_adult_home = model.compartment_names[3]
    assert model.get_infection_density_multipier(s_child_work) == work_density
    assert model.get_infection_density_multipier(s_child_home) == home_density
    assert model.get_infection_density_multipier(s_adult_work) == work_density
    assert model.get_infection_density_multipier(s_adult_home) == home_density
    assert model.get_infection_frequency_multipier(s_child_work) == work_freq
    assert model.get_infection_frequency_multipier(s_child_home) == home_freq
    assert model.get_infection_frequency_multipier(s_adult_work) == work_freq
    assert model.get_infection_frequency_multipier(s_adult_home) == home_freq


def test_strat_get_infection_multipier__with_double_strat_and_both_strats_mixing():
    """
    Check FoI when a two 2-strata stratification applied and both stratifications have a mixing matrix.
    """
    # Create the model
    model = StratifiedModel(**MODEL_KWARGS)
    am = np.array([[2, 3], [5, 7]])
    model.stratify(
        stratification_name="agegroup",
        strata_request=["child", "adult"],
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={"child": 0.3, "adult": 0.7},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=am,
    )
    lm = np.array([[11, 13], [17, 19]])
    model.stratify(
        stratification_name="location",
        strata_request=["work", "home"],  # These kids have jobs.
        compartments_to_stratify=["S", "I", "R"],
        comp_split_props={},
        flow_adjustments={},
        infectiousness_adjustments={},
        mixing_matrix=lm,
    )
    expected_mixing = np.array(
        [
            [2 * 11, 2 * 13, 3 * 11, 3 * 13],
            [2 * 17, 2 * 19, 3 * 17, 3 * 19],
            [5 * 11, 5 * 13, 7 * 11, 7 * 13],
            [5 * 17, 5 * 19, 7 * 17, 7 * 19],
        ]
    )
    assert_array_equal(model._static_mixing_matrix, expected_mixing)
    assert model.mixing_categories == [
        {"agegroup": "child", "location": "work"},
        {"agegroup": "child", "location": "home"},
        {"agegroup": "adult", "location": "work"},
        {"agegroup": "adult", "location": "home"},
    ]
    assert model.compartment_names == [
        "SXagegroup_childXlocation_work",
        "SXagegroup_childXlocation_home",
        "SXagegroup_adultXlocation_work",
        "SXagegroup_adultXlocation_home",
        "IXagegroup_childXlocation_work",
        "IXagegroup_childXlocation_home",
        "IXagegroup_adultXlocation_work",
        "IXagegroup_adultXlocation_home",
        "RXagegroup_childXlocation_work",
        "RXagegroup_childXlocation_home",
        "RXagegroup_adultXlocation_work",
        "RXagegroup_adultXlocation_home",
    ]
    expected_comp_vals = np.array([148.5, 148.5, 346.5, 346.5, 1.5, 1.5, 3.5, 3.5, 0, 0, 0, 0])
    assert_array_equal(model.compartment_values, expected_comp_vals)

    # Do pre-run FoI calcs.
    model.prepare_to_run()
    exp_lookup = {
        0: 0,  # SXagegroup_childXlocation_work -> children at work
        1: 1,  # SXagegroup_childXlocation_home -> children at home
        2: 2,  # SXagegroup_adultXlocation_work -> adults at work
        3: 3,  # SXagegroup_adultXlocation_home -> adults at home
        4: 0,  # IXagegroup_childXlocation_work -> children at work
        5: 1,  # IXagegroup_childXlocation_home -> children at home
        6: 2,  # IXagegroup_adultXlocation_work -> adults at work
        7: 3,  # IXagegroup_adultXlocation_home -> adults at home
        8: 0,  # RXagegroup_childXlocation_work -> children at work
        9: 1,  # RXagegroup_childXlocation_home -> children at home
        10: 2,  # RXagegroup_adultXlocation_work -> adults at work
        11: 3,  # RXagegroup_adultXlocation_home -> adults at home
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
    assert_array_equal(model.infectiousness_multipliers, exp_mults)
    assert_array_equal(model.category_matrix, exp_matrix)
    assert model.category_lookup == exp_lookup

    # Do pre-iteration FoI calcs
    model.prepare_time_step(0, model.compartment_values)
    exp_pops = np.array(
        [
            [150],  # children at work
            [150],  # children at home
            [350],  # adults at work
            [350],  # adults at home
        ]
    )
    assert_array_equal(model.category_populations, exp_pops)
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
    assert_array_equal(model.infection_density, exp_density)
    assert_allclose(model.infection_frequency, exp_frequency, rtol=0, atol=1e-9)

    # Get multipliers
    s_child_work = model.compartment_names[0]
    s_child_home = model.compartment_names[1]
    s_adult_work = model.compartment_names[2]
    s_adult_home = model.compartment_names[3]
    assert abs(model.get_infection_density_multipier(s_child_work) - child_work_density) <= 1e-9
    assert abs(model.get_infection_density_multipier(s_child_home) - child_home_density) <= 1e-9
    assert abs(model.get_infection_density_multipier(s_adult_work) - adult_work_density) <= 1e-9
    assert abs(model.get_infection_density_multipier(s_adult_home) - adult_home_density) <= 1e-9
    assert abs(model.get_infection_frequency_multipier(s_child_work) - child_work_freq) <= 1e-9
    assert abs(model.get_infection_frequency_multipier(s_child_home) - child_home_freq) <= 1e-9
    assert abs(model.get_infection_frequency_multipier(s_adult_work) - adult_work_freq) <= 1e-9
    assert abs(model.get_infection_frequency_multipier(s_adult_home) - adult_home_freq) <= 1e-9
