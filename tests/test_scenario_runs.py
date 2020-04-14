"""
Test running of multiple model scenarios
"""
import pytest
import numpy as np
from unittest import mock

from autumn.tool_kit import run_multi_scenario, initialise_scenario_run
from autumn.tool_kit.scenarios import get_scenario_start_index


@mock.patch("autumn.tool_kit.scenarios.initialise_scenario_run", autospec=True)
def test_run_multi_scenario(mock_init_scenario_run):
    """
    Ensure that `run_multi_scenario` builds the models correctly and then runs the models.
    """
    # Set up a mock model_builder to build a baseline model
    mock_baseline_model = mock.Mock()
    mock_model_builder = mock.Mock()
    mock_model_builder.return_value = mock_baseline_model

    # Set up a mock scenario model builder
    mock_scenario_1_model = mock.Mock()
    mock_scenario_2_model = mock.Mock()
    mock_init_scenario_run.side_effect = [mock_scenario_1_model, mock_scenario_2_model]

    # Run the multiple scenarios
    start_time = 1990
    param_lookup = {
        0: {"start_time": 1989, "contact_rate": 10},
        1: {"start_time": 1992, "contact_rate": 11},
        2: {"start_time": 1993, "contact_rate": 12},
    }
    params = {'scenario_start_time': start_time, 'default': param_lookup[0]}
    models = run_multi_scenario(param_lookup, params, mock_model_builder)

    # Ensure models were returned
    assert models == [mock_baseline_model, mock_scenario_1_model, mock_scenario_2_model]

    # Ensure models were build with correct parameters
    mock_model_builder.assert_called_once_with({"start_time": 1989, "contact_rate": 10})
    mock_init_scenario_run.assert_has_calls(
        [
            mock.call(
                mock_baseline_model, {"start_time": 1990, "contact_rate": 11}, mock_model_builder
            ),
            mock.call(
                mock_baseline_model, {"start_time": 1990, "contact_rate": 12}, mock_model_builder
            ),
        ]
    )

    # Ensure models were run
    mock_baseline_model.run_model.assert_called_once()
    mock_scenario_1_model.run_model.assert_called_once()
    mock_scenario_2_model.run_model.assert_called_once()


def test_initialise_scenario_run():
    """
    Ensure initialize scenario run creates a new scenario model from the
    baseline model with the correct start time and initial compartment values.
    """
    # Create a mock model builder function
    mock_scenario_model = mock.Mock()
    mock_model_builder = mock.Mock()
    mock_model_builder.return_value = mock_scenario_model
    # Create a mock baseline model, which has already been run.
    mock_baseline_model = mock.Mock()
    mock_baseline_model.times = [1990, 1991, 1992, 1993, 1994, 1995]
    mock_baseline_model.outputs = np.array(
        [[50, 50], [51, 49], [52, 48], [53, 47], [54, 46], [55, 45]]
    )
    # Run the initialisation
    scenario_params = {"start_time": 1991.5, "contact_rate": 14}
    scenario_model = initialise_scenario_run(
        mock_baseline_model, scenario_params, mock_model_builder
    )
    # Check that we added the correct start time for the scenario model
    assert scenario_params == {"start_time": 1991, "contact_rate": 14}
    # Assert that we created a new scenario model with model builder
    mock_model_builder.assert_called_once_with(scenario_params)
    assert scenario_model is mock_scenario_model
    assert (scenario_model.compartment_values == np.array([51, 49])).all()


START_INDEX_TEST_CASES = [
    # times, start, index
    [[1, 2, 3, 4, 5], 2, 1],  # Works in happy case
    [[1, 2, 3, 4, 5], 1, 0],  # Works at first time
    [[1, 2, 3, 4, 5], 0, 0],  # Works before first time
    [[1, 2, 3, 4, 5], 4, 3],  # Works at penultimate time
    [[1, 2, 3, 4, 5], 4.5, 3],  # Works in between times
]


@pytest.mark.parametrize("times,start,idx", START_INDEX_TEST_CASES)
def test_get_scenario_start_index(times, start, idx):
    """
    Ensure the correct timestep index is selected for the scenario start.
    """
    assert get_scenario_start_index(times, start) == idx


START_INDEX_INVALID_TEST_CASES = [
    [[1, 2, 3, 4, 5], 5],  # The last timestep
    [[1, 2, 3, 4, 5], 6],  # After the last timestep
]


@pytest.mark.parametrize("times,start", START_INDEX_INVALID_TEST_CASES)
def test_get_scenario_start_index__with_bad_start_times__expect_failure(times, start):
    """
    Ensure a validation error is raised when the start time is invalid.
    """
    with pytest.raises(ValueError):
        get_scenario_start_index(times, start)
