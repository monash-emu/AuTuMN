"""
Test the "extra flows" feature - where we add more flows after stratifying the model compartments.
"""
import numpy as np

from summer.flow import InfectionFrequencyFlow
from summer.model import StratifiedModel
from summer.constants import Flow, BirthApproach


def test_add_extra_flows__before_strat_then_stratify():
    """
    Ensure that adding an extra flow without any stratifications creates a new flow.
    Then check that stratifying the model creates the new stratified flows correctly.
    """
    # Create the model
    model = StratifiedModel(
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        compartment_names=["S", "I", "R"],
        initial_conditions={"S": 100},
        parameters={},
        requested_flows=[],
        starting_population=1000,
        infectious_compartments=["I"],
        entry_compartment="S",
        birth_approach=BirthApproach.NO_BIRTH,
    )
    # Add a new flow.
    model.add_extra_flow(
        flow={
            "type": Flow.INFECTION_FREQUENCY,
            "origin": "S",
            "to": "I",
            "parameter": "contact_rate",
        },
        source_strata={},
        dest_strata={},
        expected_flow_count=1,
    )
    # Ensure the flow was added correctly.
    assert len(model.flows) == 1
    flow = model.flows[0]
    assert type(flow) is InfectionFrequencyFlow
    assert flow.source == "S"
    assert flow.dest == "I"
    assert flow.adjustments == []
    assert flow.param_name == "contact_rate"
    # Stratify the model.
    model.stratify(
        "agegroup", strata_request=["young", "old"], compartments_to_stratify=["S", "I", "R"]
    )
    # Ensure the new flow was stratified correctly into two flows..
    assert len(model.flows) == 2
    flow_1 = model.flows[0]
    assert type(flow_1) is InfectionFrequencyFlow
    assert flow_1.source == "SXagegroup_young"
    assert flow_1.dest == "IXagegroup_young"
    assert flow_1.adjustments == []
    assert flow_1.param_name == "contact_rate"
    flow_2 = model.flows[1]
    assert type(flow_2) is InfectionFrequencyFlow
    assert flow_2.source == "SXagegroup_old"
    assert flow_2.dest == "IXagegroup_old"
    assert flow_2.adjustments == []
    assert flow_2.param_name == "contact_rate"


def test_add_extra_flows__after_single_strat():
    """
    Ensure that adding an extra flow with an existing stratification creates a new flow.
    This should produce the exact same result as `test_add_extra_flows__before_strat_then_stratify`.
    """
    # Create the model
    model = StratifiedModel(
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        compartment_names=["S", "I", "R"],
        initial_conditions={"S": 100},
        parameters={},
        requested_flows=[],
        starting_population=1000,
        infectious_compartments=["I"],
        entry_compartment="S",
        birth_approach=BirthApproach.NO_BIRTH,
    )
    # Ensure there are no flows yet.
    assert len(model.flows) == 0
    # Stratify the model.
    model.stratify(
        "agegroup", strata_request=["young", "old"], compartments_to_stratify=["S", "I", "R"]
    )
    # Add a new flow.
    model.add_extra_flow(
        flow={
            "type": Flow.INFECTION_FREQUENCY,
            "origin": "S",
            "to": "I",
            "parameter": "contact_rate",
        },
        source_strata={},
        dest_strata={},
        expected_flow_count=2,
    )
    # Ensure the new flow was stratified correctly into two flows..
    assert len(model.flows) == 2
    flow_1 = model.flows[0]
    assert type(flow_1) is InfectionFrequencyFlow
    assert flow_1.source == "SXagegroup_young"
    assert flow_1.dest == "IXagegroup_young"
    assert flow_1.adjustments == []
    assert flow_1.param_name == "contact_rate"
    flow_2 = model.flows[1]
    assert type(flow_2) is InfectionFrequencyFlow
    assert flow_2.source == "SXagegroup_old"
    assert flow_2.dest == "IXagegroup_old"
    assert flow_2.adjustments == []
    assert flow_2.param_name == "contact_rate"


def test_add_extra_flows__after_single_strat_with_cross_cut():
    """
    Ensure that adding an extra flow with an existing stratification creates a new flow.
    This flow cuts across strata.
    """
    # Create the model
    model = StratifiedModel(
        times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        compartment_names=["S", "I", "R"],
        initial_conditions={"S": 100},
        parameters={},
        requested_flows=[],
        starting_population=1000,
        infectious_compartments=["I"],
        entry_compartment="S",
        birth_approach=BirthApproach.NO_BIRTH,
    )
    # Ensure there are no flows yet.
    assert len(model.flows) == 0
    # Stratify the model.
    model.stratify(
        "agegroup", strata_request=["young", "old"], compartments_to_stratify=["S", "I", "R"]
    )
    # Add a new flow.
    model.add_extra_flow(
        flow={
            "type": Flow.INFECTION_FREQUENCY,
            "origin": "S",
            "to": "I",
            "parameter": "contact_rate",
        },
        source_strata={"agegroup": "young"},
        dest_strata={"agegroup": "old"},
        expected_flow_count=1,
    )
    # Ensure the new flow was added correctly.
    assert len(model.flows) == 1
    flow = model.flows[0]
    assert type(flow) is InfectionFrequencyFlow
    assert flow.source == "SXagegroup_young"
    assert flow.dest == "IXagegroup_old"
    assert flow.adjustments == []
    assert flow.param_name == "contact_rate"
    # Stratify the model again.
    model.stratify(
        "location", strata_request=["urban", "rural"], compartments_to_stratify=["S", "I", "R"]
    )
    # Ensure the new flow was stratified correctly into two flows.
    assert len(model.flows) == 2
    flow_1 = model.flows[0]
    assert type(flow_1) is InfectionFrequencyFlow
    assert flow_1.source == "SXagegroup_youngXlocation_urban"
    assert flow_1.dest == "IXagegroup_oldXlocation_urban"
    assert flow_1.adjustments == []
    assert flow_1.param_name == "contact_rate"
    flow_2 = model.flows[1]
    assert type(flow_2) is InfectionFrequencyFlow
    assert flow_2.source == "SXagegroup_youngXlocation_rural"
    assert flow_2.dest == "IXagegroup_oldXlocation_rural"
    assert flow_2.adjustments == []
    assert flow_2.param_name == "contact_rate"