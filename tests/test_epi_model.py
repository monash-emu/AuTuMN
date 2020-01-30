"""
End-to-end tests for the EpiModel - a disease agnostic compartmental model from SUMMER
"""
import numpy as np

from summer_py.summer_model import EpiModel
from autumn.constants import Compartment, Flow
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
    assert len(model.outputs) == 6
    assert (model.outputs == [100.0, 0.0]).all()


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
    assert len(model.outputs) == 6
    assert (model.outputs == [100.0, 0.0]).all()


def test_epi_model__with_infection_frequency__expect_all_infected():
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected, transmission highly likely.
    pop = 100
    model = EpiModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
        initial_conditions={Compartment.INFECTIOUS: 1},
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
    # Expect that everyone gets infected very quickly.
    assert np.round(model.outputs) == np.array(
        [[99.0, 1.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]
    )


def test_epi_model__with_infection_density__expect_all_infected():
    pass


def test_epi_model__with_birth_rate__expect_pop_increase():
    pass


def test_epi_model__with_death_rate__expect_pop_decrease():
    pass


def test_epi_model__with_birth_and_death_rate__expect_NOT_SURE_YET():
    pass
