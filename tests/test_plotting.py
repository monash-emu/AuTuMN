from tempfile import TemporaryDirectory

from summer.model import StratifiedModel

from autumn.plots import plot_scenarios
from autumn.tool_kit import Scenario, get_integration_times
from autumn.constants import BirthApproach


def test_for_smoke__plot_scenarios():
    """
    Smoke test plot_scenarios to ensure it runs without crashing.
    Does not test all code execution paths - eg. no generated outputs.
    """
    plot_config = {
        "translations": {},
        "outputs_to_plot": [],
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
        compartment_names=["S", "I"],
        initial_conditions={"S": pop},
        parameters={},
        requested_flows=[],
        starting_population=pop,
        infectious_compartments=["I"],
        birth_approach=BirthApproach.NO_BIRTH,
        entry_compartment="S",
    )
    # Add basic age stratification
    model.stratify("age", strata_request=[0, 5, 15, 60], compartments_to_stratify=["S", "I"])
    return model
