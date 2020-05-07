from tempfile import TemporaryDirectory

from summer.model import StratifiedModel

from autumn.outputs import plot_scenarios
from autumn.tool_kit import Scenario, get_integration_times
from autumn.constants import Compartment, Stratification


def test_for_smoke__plot_scenarios():
    """
    Smoke test plot_scenarios to ensure it runs without crashing.
    Does not test all code execution paths - eg. no generated outputs.
    """
    plot_config = {
        "translations": {},
        "outputs_to_plot": [],
        "pop_distribution_strata": [],
        "prevalence_combos": [],
        "input_function": {"start_time": 0, "func_names": []},
        "parameter_category_values": {"time": 0, "param_names": []},
    }
    # Build and run scenarios
    params = {"default": {}, "scenario_start_time": 2002, "scenarios": {1: {}}}
    scenarios = [
        Scenario(_build_model, 0, params),
        Scenario(_build_model, 1, params),
    ]
    scenarios[0].run()
    scenarios[1].run(base_model=scenarios[0].model)
    with TemporaryDirectory() as tmp_out_dir:
        plot_scenarios(scenarios, tmp_out_dir, plot_config)


def _build_model(*args, **kwargs):
    pop = 1000
    model = StratifiedModel(
        times=get_integration_times(2000, 2005, 1),
        compartment_types=[Compartment.SUSCEPTIBLE, Compartment.INFECTIOUS],
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
    return model
