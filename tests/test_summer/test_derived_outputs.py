import numpy as np
from numpy.testing import assert_array_equal

from summer.model import StratifiedModel
from summer.constants import Flow, BirthApproach
from summer.model.derived_outputs import (
    DerivedOutputCalculator,
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)


def test_calculate_derived_outputs__with_no_outputs_requested():
    calc = DerivedOutputCalculator()
    model = _get_model()
    derived_outputs = calc.calculate(model)
    derived_outputs.keys() == []


def test_calculate_derived_outputs__with_no_strata_requested():
    flow_outputs = {
        "infection": TransitionFlowOutput(source="S", dest="I", source_strata={}, dest_strata={}),
        "recovery": TransitionFlowOutput(source="I", dest="R", source_strata={}, dest_strata={}),
        "infect_death": InfectionDeathFlowOutput(source="I", source_strata={}),
    }

    def _func(time_idx, model, compartment_values, derived_outputs):
        return derived_outputs["infect_death"][time_idx] + derived_outputs["recovery"][time_idx]

    calc = DerivedOutputCalculator()
    calc.add_flow_derived_outputs(flow_outputs)
    calc.add_function_derived_outputs({"infect_exit": _func})

    model = _get_model()
    derived_outputs = calc.calculate(model)
    derived_outputs.keys() == ["infection", "recovery", "infect_death", "infect_exit"]
    assert_array_equal(derived_outputs["infect_death"], np.array([0, 200, 200, 300, 200, 300]))
    assert_array_equal(derived_outputs["recovery"], np.array([0, 120, 120, 180, 120, 180]))
    assert_array_equal(derived_outputs["infect_exit"], np.array([0, 320, 320, 480, 320, 480]))
    assert_array_equal(derived_outputs["infection"], np.array([0, 24, 16, 12, 8, 0]))


def test_calculate_derived_outputs__with_strata_requested():
    flow_outputs = {
        "infection": TransitionFlowOutput(
            source="S", dest="I", source_strata={}, dest_strata={"agegroup": "0"}
        ),
        "recovery": TransitionFlowOutput(
            source="I", dest="R", source_strata={"agegroup": "15"}, dest_strata={}
        ),
        "infect_death": InfectionDeathFlowOutput(source="I", source_strata={"agegroup": "60"}),
    }

    def _func(time_idx, model, compartment_values, derived_outputs):
        return derived_outputs["infect_death"][time_idx] + derived_outputs["recovery"][time_idx]

    calc = DerivedOutputCalculator()
    calc.add_flow_derived_outputs(flow_outputs)
    calc.add_function_derived_outputs({"infect_exit": _func})

    model = _get_model()
    derived_outputs = calc.calculate(model)
    derived_outputs.keys() == ["infection", "recovery", "infect_death", "infect_exit"]
    assert_array_equal(derived_outputs["infection"], np.array([0, 6, 4, 3, 2, 0]))
    assert_array_equal(derived_outputs["infect_death"], np.array([0, 50, 50, 75, 50, 75]))
    assert_array_equal(derived_outputs["recovery"], np.array([0, 30, 30, 45, 30, 45]))
    assert_array_equal(derived_outputs["infect_exit"], np.array([0, 80, 80, 120, 80, 120]))


def test_filter_transition_flow():
    model = _get_model()
    # Check S -> I
    output = TransitionFlowOutput(source="S", dest="I", source_strata={}, dest_strata={})
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    flow_sources = [f.source for f in flows]
    assert flow_types == [Flow.INFECTION_FREQUENCY] * 4
    assert flow_sources == ["SXagegroup_0", "SXagegroup_5", "SXagegroup_15", "SXagegroup_60"]
    # Check I -> R
    output = TransitionFlowOutput(source="I", dest="R", source_strata={}, dest_strata={})
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    flow_sources = [f.source for f in flows]
    assert flow_types == [Flow.STANDARD] * 4
    assert flow_sources == ["IXagegroup_0", "IXagegroup_5", "IXagegroup_15", "IXagegroup_60"]

    # Check S -> I with only source strata from agegroup 5
    output = TransitionFlowOutput(
        source="S", dest="I", source_strata={"agegroup": "5"}, dest_strata={}
    )
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    assert flow_types == [Flow.INFECTION_FREQUENCY] * 1
    flow_sources = [f.source for f in flows]
    assert flow_sources == ["SXagegroup_5"]

    # Check S -> I with only dest strata from agegroup 15
    output = TransitionFlowOutput(
        source="S", dest="I", source_strata={}, dest_strata={"agegroup": "15"}
    )
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    assert flow_types == [Flow.INFECTION_FREQUENCY] * 1
    flow_sources = [f.source for f in flows]
    assert flow_sources == ["SXagegroup_15"]


def test_filter_death_flows():
    model = _get_model()

    # Check S -> dead, expect no results
    output = InfectionDeathFlowOutput(source="S", source_strata={})
    assert not output.filter_flows(model.flows)

    # Check R -> dead, expect no results
    output = InfectionDeathFlowOutput(source="R", source_strata={})
    assert not output.filter_flows(model.flows)

    # Check I -> dead, expect all strata
    output = InfectionDeathFlowOutput(source="I", source_strata={})
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    flow_sources = [f.source for f in flows]
    assert flow_types == [Flow.DEATH] * 4
    assert flow_sources == ["IXagegroup_0", "IXagegroup_5", "IXagegroup_15", "IXagegroup_60"]

    # Check I -> dead, with only source strata from agegroup 5
    output = InfectionDeathFlowOutput(source="I", source_strata={"agegroup": "5"})
    flows = output.filter_flows(model.flows)
    flow_types = [f.type for f in flows]
    flow_sources = [f.source for f in flows]
    assert flow_types == [Flow.DEATH] * 1
    assert flow_sources == ["IXagegroup_5"]


def _get_model():
    pop = 1000
    params = {
        "contact_rate": 0.1,
        "recovery_rate": 0.3,
        "infect_death": 0.5,
        "crude_birth_rate": 0.7,
        "universal_death_rate": 0.11,
    }
    flows = [
        {"type": Flow.INFECTION_FREQUENCY, "origin": "S", "to": "I", "parameter": "contact_rate"},
        {"type": Flow.STANDARD, "to": "R", "origin": "I", "parameter": "recovery_rate"},
        {"type": Flow.DEATH, "origin": "I", "parameter": "infect_death"},
    ]
    model = StratifiedModel(
        times=np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        compartment_names=["S", "I", "R"],
        initial_conditions={"S": pop},
        parameters=params,
        requested_flows=flows,
        starting_population=pop,
        infectious_compartments=["I"],
        birth_approach=BirthApproach.ADD_CRUDE,
        entry_compartment="S",
    )
    # Add basic age stratification
    model.stratify("agegroup", strata_request=[0, 5, 15, 60], compartments_to_stratify=["S", "I"])
    # Pretend we ran the model
    model.prepare_to_run()
    model.outputs = np.array(
        [
            [250, 250, 250, 250, 0, 0, 0, 0, 0],
            [150, 150, 150, 150, 100, 100, 100, 100, 0],
            [100, 100, 100, 100, 100, 100, 100, 100, 200],
            [50, 50, 50, 50, 150, 150, 150, 150, 200],
            [50, 50, 50, 50, 100, 100, 100, 100, 400],
            [0, 0, 0, 0, 150, 150, 150, 150, 400],
        ]
    )
    return model
