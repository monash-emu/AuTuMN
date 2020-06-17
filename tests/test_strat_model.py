"""
End-to-end tests for the StratifiedModel - a disease agnostic compartmental model from SUMMER
"""
import pytest
import numpy as np

from summer.model import StratifiedModel
from autumn.constants import (
    Compartment,
    Flow,
    BirthApproach,
    Stratification,
    IntegrationType,
)
from autumn.tool_kit import get_integration_times


def test_strat_model__with_age__expect_ageing():
    """
    Ensure that a module with age stratification produces ageing flows,
    and the correct output.
    """
    pop = 1000
    model = StratifiedModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
    )
    # Add basic age stratification
    model.stratify(
        Stratification.AGE,
        strata_request=[0, 5, 15, 60],
        compartment_types_to_stratify=[],
        requested_proportions={},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to generally get older, but no one should die or get sick
    expected_output = [
        [250.0, 250.0, 250.0, 250.0, 0.0, 0.0, 0.0, 0.0],
        [205.0, 269.0, 270.0, 256.0, 0.0, 0.0, 0.0, 0.0],
        [168.0, 279.0, 291.0, 262.0, 0.0, 0.0, 0.0, 0.0],
        [137.0, 281.0, 313.0, 269.0, 0.0, 0.0, 0.0, 0.0],
        [112.0, 278.0, 334.0, 276.0, 0.0, 0.0, 0.0, 0.0],
        [92.0, 271.0, 354.0, 284.0, 0.0, 0.0, 0.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_strat_model__with_age_and_starting_proportion__expect_ageing():
    """
    Ensure that a module with age stratification and starting proporptions
    produces ageing flows, and the correct output.
    """
    pop = 1000
    model = StratifiedModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
    )
    # Add basic age stratification
    model.stratify(
        Stratification.AGE,
        strata_request=[0, 5, 15, 60],
        compartment_types_to_stratify=[],
        requested_proportions={"0": 0.8, "5": 0.1, "15": 0.1},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to generally get older, but no one should die or get sick.
    # Expect initial distribution of ages to be set according to "requested_proportions".
    expected_output = [
        [800.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [655.0, 228.0, 114.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [536.0, 319.0, 139.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        [439.0, 381.0, 171.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        [360.0, 421.0, 207.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        [294.0, 442.0, 245.0, 18.0, 0.0, 0.0, 0.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


def test_strat_model__with_locations__expect_no_change():
    """
    Ensure that a module with location stratification populates locations correctly.
    """
    pop = 1000
    model = StratifiedModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
    )
    # Add basic location stratification
    model.stratify(
        Stratification.LOCATION,
        strata_request=["rural", "urban", "prison"],
        compartment_types_to_stratify=[],
        requested_proportions={"rural": 0.44, "urban": 0.55, "prison": 0.01},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to start in their locations, then nothing should change,
    expected_output = [
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
        [440.0, 550.0, 10.0, 0.0, 0.0, 0.0],
    ]
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()


@pytest.mark.xfail(reason="I don't know how to implement this properly.")
def test_strat_model__with_age_and_infectiousness__expect_age_based_infectiousness():
    """
    Ensure that a module with age stratification produces ageing flows,
    and the correct output. Ensure that age-speific mortality is used.
    """
    pop = 1000
    model = StratifiedModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.EARLY_INFECTIOUS],
        initial_conditions={Compartment.SUSCEPTIBLE: pop},
        parameters={"contact_rate": None},
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
    # Add age stratification
    model.stratify(
        Stratification.AGE,
        strata_request=[0, 5, 15, 60],
        compartment_types_to_stratify=[],
        requested_proportions={},
        adjustment_requests={"contact_rate": {"0": 0.0, "5": 3.0, "15": 0.0, "60": 0.0}},
    )
    # Run the model for 5 years.
    model.run_model(integration_type=IntegrationType.ODE_INT)

    # Expect everyone to generally get older, but no one should die or get sick
    expected_output = []
    actual_output = np.round(model.outputs)
    assert (actual_output == np.array(expected_output)).all()
