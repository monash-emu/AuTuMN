"""
End-to-end tests for the EpiModel - a disease agnostic compartmental model from SUMMER
"""
import numpy as np

from summer_py.summer_model import EpiModel
from autumn.constants import Compartment, Flow, BirthApproach
from autumn.tool_kit import get_integration_times


def test_epi_model__with_static_dynamics__expect_no_change():
    """
    Ensure that a model with two compartments and no internal dynamics results in no change.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_output = [
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_epi_model__with_birth_rate__expect_pop_increase():
    """
    Ensure that a model with two compartments and only birth rate dynamics results in more people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some babies at ~2 babies / 100 / year.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"crude_birth_rate": 2e-2},
        requested_flows=[],
        birth_approach=BirthApproach.ADD_CRUDE,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that we have more people in the population
    expected_output = [
        [[100.0, 0.0], [102.0, 0.0], [104.0, 0.0], [106.0, 0.0], [108.0, 0.0], [111.0, 0.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_epi_model__with_death_rate__expect_pop_decrease():
    """
    Ensure that a model with two compartments and only death rate dynamics results in fewer people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some dying at ~2 people / 100 / year.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"universal_death_rate": 2e-2},
        requested_flows=[],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that we have more people in the population
    expected_output = [
        [100.0, 0.0],
        [98.0, 0.0],
        [96.0, 0.0],
        [94.0, 0.0],
        [92.0, 0.0],
        [91.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_epi_model__with_infect_death_rate__expect_infected_pop_decrease():
    pass


def test_epi_model__with_no_infected__expect_no_change():
    """
    Ensure that if no one has the disease, then no one gets the disease in the future.
    """
    # Set up a model with 100 people, all susceptible, transmission highly likely, but no one is infected.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"contact_rate": 10},
        requested_flows=[
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_output = [
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_epi_model__with_infection_frequency__expect_all_infected():
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.INFECTIOUS: 1},
        parameters={"contact_rate": 3},
        requested_flows=[
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that everyone gets infected eventually.
    expected_output = [
        [[99.0, 1.0], [83.0, 17.0], [20.0, 80.0], [1.0, 99.0], [0.0, 100.0], [0.0, 100.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_epi_model__with_infection_density__expect_all_infected():
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.INFECTIOUS: 1},
        parameters={"contact_rate": 0.03},
        requested_flows=[
            {
                "type": Flow.INFECTION_DENSITY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model()
    # Expect that everyone gets infected eventually.
    expected_output = [
        [[99.0, 1.0], [83.0, 17.0], [20.0, 80.0], [1.0, 99.0], [0.0, 100.0], [0.0, 100.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()
