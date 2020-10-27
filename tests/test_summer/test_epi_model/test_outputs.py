"""
End-to-end tests for the EpiModel and StratifiedModel - a disease agnostic compartmental model from SUMMER

We expect the StratifiedModel and EpiModel to work the same in these basic cases.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from summer.model import EpiModel, StratifiedModel
from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    IntegrationType,
)


def _get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return np.linspace(start_year, end_year, n_iter)


MODEL_KWARGS = {
    "times": _get_integration_times(2000, 2005, 1),
    "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
    "initial_conditions": {Compartment.EARLY_INFECTIOUS: 100},
    "parameters": {},
    "requested_flows": [],
    "starting_population": 200,
    "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
    "birth_approach": BirthApproach.NO_BIRTH,
    "entry_compartment": Compartment.SUSCEPTIBLE,
}


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_static_dynamics__expect_no_change(ModelClass):
    """
    Ensure that a model with two compartments and no internal dynamics results in no change.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    model_kwargs = {}
    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ]
    )
    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, rtol=0, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_birth_rate__expect_pop_increase(ModelClass):
    """
    Ensure that a model with two compartments and only birth rate dynamics results in more people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some babies at ~2 babies / 100 / year.
    model_kwargs = {
        "parameters": {"crude_birth_rate": 0.02},
        "birth_approach": BirthApproach.ADD_CRUDE,
    }
    # Expect that we have more people in the population per year
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [104.0, 100.0],
            [108.2, 100.0],
            [112.4, 100.0],
            [116.7, 100.0],
            [121.0, 100.0],
        ]
    )

    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_death_rate__expect_pop_decrease(ModelClass):
    """
    Ensure that a model with two compartments and only death rate dynamics results in fewer people.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some dying at ~2 people / 100 / year.
    model_kwargs = {
        "parameters": {"universal_death_rate": 0.02},
        "birth_approach": BirthApproach.NO_BIRTH,
    }
    # Expect that we have fewer people in the population per year
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [98.0, 98.0],
            [96.1, 96.1],
            [94.2, 94.2],
            [92.3, 92.3],
            [90.5, 90.5],
        ]
    )
    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_birth_and_death_rate__expect_pop_static_overall(ModelClass):
    model_kwargs = {
        "parameters": {"universal_death_rate": 0.02, "crude_birth_rate": 0.02},
        "birth_approach": BirthApproach.ADD_CRUDE,
    }
    # Expect that we have fewer people in the population per year
    # Results modelled using 1e-6 timestep, small tweaks in favour of summer model.
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [102.0, 98.0],
            [104.0, 96.0],
            [105.8, 94.2],  # Tweaked.
            [107.7, 92.3],  # Tweaked.
            [109.5, 90.5],  # Tweaked.
        ]
    )
    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_birth_and_death_rate_replace_deaths__expect_pop_static_overall(ModelClass):
    model_kwargs = {
        "parameters": {"universal_death_rate": 0.02},
        "birth_approach": BirthApproach.REPLACE_DEATHS,
    }
    # Expect that we have fewer people in the population per year
    # Results modelled using 1e-6 timestep, small tweaks in favour of summer model.
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [102.0, 98.0],
            [104.0, 96.0],
            [105.8, 94.2],  # Tweaked.
            [107.7, 92.3],  # Tweaked.
            [109.5, 90.5],  # Tweaked.
        ]
    )
    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_higher_birth_than_and_death_rate__expect_pop_increase(ModelClass):
    model_kwargs = {
        "parameters": {"universal_death_rate": 0.02, "crude_birth_rate": 0.1},
        "birth_approach": BirthApproach.ADD_CRUDE,
    }
    # Expect that we have more people in the population per year
    # Results modelled using 1e-6 timestep, small tweaks in favour of summer model.
    expected_arr = np.array(
        [
            [100.0, 100.0],  # Initial conditions
            [118.6, 98.0],  # Tweaked ~0.1
            [138.6, 96.1],  # Tweaked ~0.4
            [160.1, 94.2],  # Tweaked ~0.9
            [183.1, 92.3],  # Tweaked ~1.7
            [207.9, 90.5],  # Tweaked ~2.7
        ]
    )

    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.SOLVE_IVP)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_recovery_rate__expect_all_recover(ModelClass):
    """
    Ensure that a model with three compartments and only recovery dynamics
    results in (almost) everybody recovering.
    """
    # Set up a model with 100 people, all infectious.
    # Add recovery dynamics.
    pop = 100
    model_kwargs = {
        "times": _get_integration_times(2000, 2005, 1),
        "compartment_names": [
            Compartment.SUSCEPTIBLE,
            Compartment.EARLY_INFECTIOUS,
            Compartment.RECOVERED,
        ],
        "initial_conditions": {Compartment.EARLY_INFECTIOUS: pop},
        "parameters": {"recovery": 1},
        "requested_flows": [
            {
                "type": Flow.STANDARD,
                "parameter": "recovery",
                "origin": Compartment.EARLY_INFECTIOUS,
                "to": Compartment.RECOVERED,
            }
        ],
        "birth_approach": BirthApproach.NO_BIRTH,
        "starting_population": pop,
        "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
        "entry_compartment": Compartment.SUSCEPTIBLE,
    }
    # Expect that almost everyone recovers
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [0.00, 100.00, 0.00],  # Initial conditions
            [0.00, 36.79, 63.21],
            [0.00, 13.53, 86.47],
            [0.00, 4.98, 95.02],
            [0.00, 1.83, 98.17],
            [0.00, 0.67, 99.33],
        ]
    )

    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.ODE_INT)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infect_death_rate__expect_infected_pop_decrease(ModelClass):
    """
    Ensure that a model with two compartments and only infected death rate dynamics
    results in fewer infected people, but no change to susceptible pop.
    """
    # Set up a model with 100 people, all susceptible, no transmission possible.
    # Add some dying at ~2 people / 100 / year.
    pop = 100
    model_kwargs = {
        "times": _get_integration_times(2000, 2005, 1),
        "compartment_names": [Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        "initial_conditions": {Compartment.EARLY_INFECTIOUS: 50},
        "parameters": {"infect_death": 0.02},
        "requested_flows": [
            {
                "type": Flow.DEATH,
                "parameter": "infect_death",
                "origin": Compartment.EARLY_INFECTIOUS,
            }
        ],
        "birth_approach": BirthApproach.NO_BIRTH,
        "starting_population": pop,
        "infectious_compartments": [Compartment.EARLY_INFECTIOUS],
        "entry_compartment": Compartment.SUSCEPTIBLE,
    }

    # Expect that we have more people in the population
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [50.00, 50.00],  # Initial conditions
            [50.00, 49.01],
            [50.00, 48.04],
            [50.00, 47.09],
            [50.00, 46.16],
            [50.00, 45.24],
        ]
    )
    kwargs = {**MODEL_KWARGS, **model_kwargs}
    model = ModelClass(**kwargs)
    model.run_model(integration_type=IntegrationType.ODE_INT)
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_no_infected__expect_no_change(ModelClass):
    """
    Ensure that if no one has the disease, then no one gets the disease in the future.
    """
    # Set up a model with 100 people, all susceptible, transmission highly likely, but no one is infected.
    pop = 100
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
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
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect that no one has moved from sucsceptible to infections at any point in time
    expected_arr = np.array(
        [
            [100.0, 0.0],  # Initial conditions
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infection_frequency__expect_all_infected(ModelClass):
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
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
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect that everyone gets infected eventually.
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [99.00, 1.00],  # Initial conditions
            [83.13, 16.87],
            [19.70, 80.30],
            [1.21, 98.79],
            [0.06, 99.94],
            [0.00, 100.00],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_infection_density__expect_all_infected(ModelClass):
    """
    Ensure that a model with two compartments and one-way internal dynamics results in all infected.
    """
    # Set up a model with 100 people, all susceptible execept 1 infected.
    pop = 100
    model = ModelClass(
        times=_get_integration_times(2000, 2005, 1),
        compartment_names=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
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
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect that everyone gets infected eventually.
    # Results modelled using 1e-6 timestep.
    expected_arr = np.array(
        [
            [99.00, 1.00],  # Initial conditions
            [83.13, 16.87],
            [19.70, 80.30],
            [1.21, 98.79],
            [0.06, 99.94],
            [0.00, 100.00],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)


@pytest.mark.parametrize("ModelClass", [EpiModel, StratifiedModel])
def test_epi_model__with_complex_dynamics__expect_correct_outputs(ModelClass):
    """
    Ensure that a EpiModel with the "full suite" of TB dynamics produces correct results:
        - 5 compartments
        - birth rate +  universal death rate
        - standard inter-compartment flows
    """
    # Set up a model with 1000 people, 100 intially infected
    pop = 1000
    times = _get_integration_times(2000, 2005, 1)
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
        "crude_birth_rate": 0.02,  # ~ 2 babies / 100 / year
        "universal_death_rate": 0.02,  # ~ 2 deaths / 100 / year
        # Compartment flow params
        "infect_death": 0.4,
        "recovery": 0.2,
        "contact_rate": 14,
        "contact_rate_recovered": 14,
        "contact_rate_infected": 3,
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
            "parameter": "contact_rate_infected",
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
            "type": Flow.DEATH,
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
        infectious_compartments=[Compartment.EARLY_INFECTIOUS],
        entry_compartment=Compartment.SUSCEPTIBLE,
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect that the results are consistent, nothing crazy happens.
    # These results were not independently calculated, so more of an "acceptance test".
    expected_arr = np.array(
        [
            [900.0, 100.0, 0.0, 0.0, 0.0],
            [66.1, 307.2, 274.2, 203.8, 75.3],
            [2.9, 345.3, 220.5, 150.9, 69.4],
            [2.2, 297.0, 175.6, 127.3, 58.1],
            [1.8, 248.8, 145.6, 106.4, 48.5],
            [1.5, 207.8, 121.5, 88.8, 40.5],
        ]
    )
    assert_allclose(model.outputs, expected_arr, atol=0.1, verbose=True)
