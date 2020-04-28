"""
End-to-end tests for the EpiModel and StratifiedModel - a disease agnostic compartmental model from SUMMER

We expect the StratifiedModel and EpiModel to work the same in these basic cases.
"""
import pytest
import numpy as np

from summer.model import EpiModel, StratifiedModel
from autumn.constants import Compartment, Flow, BirthApproach, IntegrationType
from autumn.tool_kit import get_integration_times


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_static_dynamics__expect_no_change(ModelClass):
    """
    Ensure that a model with two compartments and no internal dynamics results in no change.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
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


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_birth_rate__expect_pop_increase(ModelClass):
    """
    Ensure that a model with two compartments and only birth rate dynamics results in more people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some babies at ~2 babies / 100 / year.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"crude_birth_rate": 2e-2},
        requested_flows=[],
        birth_approach=BirthApproach.ADD_CRUDE,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that we have more people in the population
    expected_output = [
        [[100.0, 0.0], [102.0, 0.0], [104.0, 0.0], [106.0, 0.0], [108.0, 0.0], [111.0, 0.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_death_rate__expect_pop_decrease(ModelClass):
    """
    Ensure that a model with two compartments and only death rate dynamics results in fewer people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some dying at ~2 people / 100 / year.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"universal_death_rate": 2e-2},
        requested_flows=[],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
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


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_recovery_rate__expect_all_recover(ModelClass):
    """
    Ensure that a model with three compartments and only recovery dynamics
    results in (almost) everybody recovering.
    """
    # Set up a model with 100 people, all infectious.
    # Add recovery dynamics.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[
            Compartment.SUSCEPTIBLE,
            Compartment.EARLY_INFECTIOUS,
            Compartment.RECOVERED,
        ],
        initial_conditions={Compartment.EARLY_INFECTIOUS: pop},
        parameters={"recovery": 1},
        requested_flows=[
            {
                "type": Flow.STANDARD,
                "parameter": "recovery",
                "origin": Compartment.EARLY_INFECTIOUS,
                "to": Compartment.RECOVERED,
            }
        ],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that almost everyone recovers
    expected_output = [
        [0.0, 100.0, 0.0],
        [0.0, 37.0, 63.0],
        [0.0, 14.0, 86.0],
        [0.0, 5.0, 95.0],
        [0.0, 2.0, 98.0],
        [0.0, 1.0, 99.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infect_death_rate__expect_infected_pop_decrease(ModelClass):
    """
    Ensure that a model with two compartments and only infected death rate dynamics
    results in fewer infected people, but no change to susceptible pop.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some dying at ~2 people / 100 / year.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.EARLY_INFECTIOUS: 50},
        parameters={"infect_death": 2e-2},
        requested_flows=[
            {
                "type": Flow.COMPARTMENT_DEATH,
                "parameter": "infect_death",
                "origin": Compartment.EARLY_INFECTIOUS,
            }
        ],
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that we have more people in the population
    expected_output = [
        [50.0, 50.0],
        [50.0, 49.0],
        [50.0, 48.0],
        [50.0, 47.0],
        [50.0, 46.0],
        [50.0, 45.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_no_infected__expect_no_change(ModelClass):
    """
    Ensure that if no one has the disease, then no one gets the disease in the future.
    """
    # Set up a model with 100 people, all susceptible, transmission highly likely, but no one is infected.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"contact_rate": 10},
        requested_flows=[
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
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


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infection_frequency__expect_all_infected(ModelClass):
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.EARLY_INFECTIOUS: 1},
        parameters={"contact_rate": 3},
        requested_flows=[
            {
                "type": Flow.INFECTION_FREQUENCY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that everyone gets infected eventually.
    expected_output = [
        [[99.0, 1.0], [83.0, 17.0], [20.0, 80.0], [1.0, 99.0], [0.0, 100.0], [0.0, 100.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infection_density__expect_all_infected(ModelClass):
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = ModelClass(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.EARLY_INFECTIOUS: 1},
        parameters={"contact_rate": 0.03},
        requested_flows=[
            {
                "type": Flow.INFECTION_DENSITY,
                "parameter": "contact_rate",
                "origin": Compartment.SUSCEPTIBLE,
                "to": Compartment.EARLY_INFECTIOUS,
            }
        ],
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that everyone gets infected eventually.
    expected_output = [
        [[99.0, 1.0], [83.0, 17.0], [20.0, 80.0], [1.0, 99.0], [0.0, 100.0], [0.0, 100.0]]
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_complex_dynamics__expect_correct_outputs(ModelClass):
    """
    Ensure that a EpiModel with the "full suite" of TB dynamics produces correct results:
        - 5 compartments
        - birth rate +  universal death rate
        - standard inter-compartment flows

    FIXME: Change parameter values to be vaugely realistic
    """
    # Set up a model with 1000 people, 100 intially infected
    pop = 1000
    times = get_integration_times(2000, 2005, 1)
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_LATENT,
        Compartment.EARLY_LATENT,
        Compartment.RECOVERED,
    ]
    initial_pop = {Compartment.EARLY_INFECTIOUS: 100}
    params = {
        # Global birth / death params
        "crude_birth_rate": 2e-2,  # ~ 2 babies / 100 / year
        "universal_death_rate": 2e-2,  # ~ 2 deaths / 100 / year
        # Compartment flow params
        "infect_death": 0.4,
        "recovery": 0.2,
        "contact_rate": 14,
        "contact_rate_recovered": 14,
        "contact_rate_late_latent": 3,
        "stabilisation": 3,
        "early_progression": 2,
        "late_progression": 1,
        "case_detection": 1,
    }
    flows = [
        # Susceptible becoming latent
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate",
            "origin": Compartment.SUSCEPTIBLE,
            "to": Compartment.EARLY_LATENT,
        },
        # Recovered becoming latent again.
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate_recovered",
            "origin": Compartment.RECOVERED,
            "to": Compartment.EARLY_LATENT,
        },
        # Late latent going back to early latent
        {
            "type": Flow.INFECTION_FREQUENCY,
            "parameter": "contact_rate_late_latent",
            "origin": Compartment.LATE_LATENT,
            "to": Compartment.EARLY_LATENT,
        },
        # Early progression from latent to infectious
        {
            "type": Flow.STANDARD,
            "parameter": "early_progression",
            "origin": Compartment.EARLY_LATENT,
            "to": Compartment.EARLY_INFECTIOUS,
        },
        # Transition from early to late latent
        {
            "type": Flow.STANDARD,
            "parameter": "stabilisation",
            "origin": Compartment.EARLY_LATENT,
            "to": Compartment.LATE_LATENT,
        },
        # Late latent becoming infectious.
        {
            "type": Flow.STANDARD,
            "parameter": "late_progression",
            "origin": Compartment.LATE_LATENT,
            "to": Compartment.EARLY_INFECTIOUS,
        },
        # Infected people dying.
        {
            "type": Flow.COMPARTMENT_DEATH,
            "parameter": "infect_death",
            "origin": Compartment.EARLY_INFECTIOUS,
        },
        # Infected people recovering naturally.
        {
            "type": Flow.STANDARD,
            "parameter": "recovery",
            "origin": Compartment.EARLY_INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
        # Infectious people recovering via manual intervention
        {
            "type": Flow.STANDARD,
            "parameter": "case_detection",
            "origin": Compartment.EARLY_INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
    ]
    model = ModelClass(
        times,
        compartments,
        initial_pop,
        params,
        flows,
        birth_approach=BirthApproach.ADD_CRUDE,
        starting_population=pop,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)
    # Expect that the results are consistent, nothing crazy happens.
    expected_output = [
        [900.0, 100.0, 0.0, 0.0, 0.0],
        [66.0, 307.0, 274.0, 204.0, 75.0],
        [3.0, 345.0, 221.0, 151.0, 69.0],
        [2.0, 297.0, 176.0, 127.0, 58.0],
        [2.0, 249.0, 146.0, 106.0, 48.0],
        [1.0, 208.0, 121.0, 89.0, 40.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()
