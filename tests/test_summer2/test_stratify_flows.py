"""
Tests to ensure that flows get stratified correctly. 
That is, when a stratification is applied, there are the right number of flows,
connected to the right compartments, with the right adjustments applied.
"""
import pytest

from summer2 import (
    CompartmentalModel,
    Stratification,
    StrainStratification,
    AgeStratification,
    Compartment as C,
)
from summer2.adjust import Multiply, Overwrite
from summer2.flows import ImportFlow, DeathFlow, CrudeBirthFlow, SojournFlow, InfectionFrequencyFlow


def test_stratify_entry_flows__with_no_explicit_adjustments():
    """
    Ensure entry flows are stratified correctly when no adjustments are requested.
    Expect flow to be conserved, split evenly over the new strata.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_importation_flow("imports", 10, "S")

    expected_flows = [ImportFlow("imports", C("S"), 10)]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    model.stratify_with(strat)

    expected_flows = [
        ImportFlow("imports", C("S", {"location": "urban"}), 10, [Multiply(0.5)]),
        ImportFlow("imports", C("S", {"location": "rural"}), 10, [Multiply(0.5)]),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    strat = Stratification("age", ["young", "old"], ["S", "I", "R"])
    model.stratify_with(strat)
    expected_flows = [
        ImportFlow(
            "imports",
            C("S", {"location": "urban", "age": "young"}),
            10,
            [Multiply(0.5), Multiply(0.5)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "urban", "age": "old"}),
            10,
            [Multiply(0.5), Multiply(0.5)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "rural", "age": "young"}),
            10,
            [Multiply(0.5), Multiply(0.5)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "rural", "age": "old"}),
            10,
            [Multiply(0.5), Multiply(0.5)],
        ),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_entry_flows__with_explicit_adjustments():
    """
    Ensure entry flows are stratified correctly when  adjustments are requested.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_importation_flow("imports", 10, "S")

    expected_flows = [ImportFlow("imports", C("S"), 10)]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    strat.add_flow_adjustments("imports", {"urban": Multiply(0.9), "rural": None})
    model.stratify_with(strat)

    expected_flows = [
        ImportFlow(
            "imports",
            C("S", {"location": "urban"}),
            10,
            [Multiply(0.9)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "rural"}),
            10,
            [],
        ),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    strat = Stratification("age", ["young", "old"], ["S", "I", "R"])
    strat.add_flow_adjustments("imports", {"young": Multiply(0.8), "old": Overwrite(1)})
    model.stratify_with(strat)
    expected_flows = [
        ImportFlow(
            "imports",
            C("S", {"location": "urban", "age": "young"}),
            10,
            [Multiply(0.9), Multiply(0.8)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "urban", "age": "old"}),
            10,
            [Multiply(0.9), Overwrite(1)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "rural", "age": "young"}),
            10,
            [Multiply(0.8)],
        ),
        ImportFlow(
            "imports",
            C("S", {"location": "rural", "age": "old"}),
            10,
            [Overwrite(1)],
        ),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_add_entry_flows_post_stratification():
    """
    Ensure we can add flows after a model is stratified.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    assert len(model._flows) == 0

    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    model.stratify_with(strat)

    with pytest.raises(AssertionError):
        model.add_importation_flow("imports", 10, "S", expected_flow_count=1)

    assert len(model._flows) == 0
    model.add_importation_flow("imports", 10, "S", expected_flow_count=2)
    assert len(model._flows) == 2

    expected_flows = [
        ImportFlow("imports", C("S", {"location": "urban"}), 10, []),
        ImportFlow("imports", C("S", {"location": "rural"}), 10, []),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_add_entry_flows_post_stratification__with_filter():
    """
    Ensure we can add flows after a model is stratified when a strata filter is applied
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )

    assert len(model._flows) == 0

    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    model.stratify_with(strat)

    assert len(model._flows) == 0
    model.add_importation_flow(
        "imports", 10, "S", dest_strata={"location": "urban"}, expected_flow_count=1
    )
    assert len(model._flows) == 1
    expected_flows = [
        ImportFlow("imports", C("S", {"location": "urban"}), 10, []),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify__age__validate_ageing_flows_added():
    """
    Ensure, when using an age stratification, that ageing flows are automatically added
    and that birth flows are all sent to age 0.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert len(model._flows) == 0
    model.add_crude_birth_flow("births", 0.02, "S")
    assert len(model._flows) == 1

    strat = AgeStratification("age", ["0", "5", "15"], ["S", "I", "R"])
    model.stratify_with(strat)

    # Expect ageing flows amongst age group and a birth flow that only goes to age 0.
    expected_flows = [
        CrudeBirthFlow("births", C("S", {"age": "0"}), 0.02),
        SojournFlow("ageing_SXage_0_to_SXage_5", C("S", {"age": "0"}), C("S", {"age": "5"}), 5),
        SojournFlow("ageing_IXage_0_to_IXage_5", C("I", {"age": "0"}), C("I", {"age": "5"}), 5),
        SojournFlow("ageing_RXage_0_to_RXage_5", C("R", {"age": "0"}), C("R", {"age": "5"}), 5),
        SojournFlow("ageing_SXage_5_to_SXage_15", C("S", {"age": "5"}), C("S", {"age": "15"}), 10),
        SojournFlow("ageing_IXage_5_to_IXage_15", C("I", {"age": "5"}), C("I", {"age": "15"}), 10),
        SojournFlow("ageing_RXage_5_to_RXage_15", C("R", {"age": "5"}), C("R", {"age": "15"}), 10),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify__age__validate_ageing_flows_added_second():
    """
    Ensure that age stratification works when applied after a previous stratification.
    """

    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert len(model._flows) == 0
    model.add_crude_birth_flow("births", 0.02, "S")
    assert len(model._flows) == 1

    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    model.stratify_with(strat)

    strat = AgeStratification("age", ["0", "5", "15"], ["S", "I", "R"])
    model.stratify_with(strat)

    # Expect ageing flows amongst age group and a birth flow that only goes to age 0.
    expected_flows = [
        CrudeBirthFlow("births", C("S", {"location": "urban", "age": "0"}), 0.02, [Multiply(0.5)]),
        CrudeBirthFlow("births", C("S", {"location": "rural", "age": "0"}), 0.02, [Multiply(0.5)]),
        SojournFlow(
            "ageing_SXlocation_urbanXage_0_to_SXlocation_urbanXage_5",
            C("S", {"location": "urban", "age": "0"}),
            C("S", {"location": "urban", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_SXlocation_ruralXage_0_to_SXlocation_ruralXage_5",
            C("S", {"location": "rural", "age": "0"}),
            C("S", {"location": "rural", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_IXlocation_urbanXage_0_to_IXlocation_urbanXage_5",
            C("I", {"location": "urban", "age": "0"}),
            C("I", {"location": "urban", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_IXlocation_ruralXage_0_to_IXlocation_ruralXage_5",
            C("I", {"location": "rural", "age": "0"}),
            C("I", {"location": "rural", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_RXlocation_urbanXage_0_to_RXlocation_urbanXage_5",
            C("R", {"location": "urban", "age": "0"}),
            C("R", {"location": "urban", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_RXlocation_ruralXage_0_to_RXlocation_ruralXage_5",
            C("R", {"location": "rural", "age": "0"}),
            C("R", {"location": "rural", "age": "5"}),
            5,
        ),
        SojournFlow(
            "ageing_SXlocation_urbanXage_5_to_SXlocation_urbanXage_15",
            C("S", {"location": "urban", "age": "5"}),
            C("S", {"location": "urban", "age": "15"}),
            10,
        ),
        SojournFlow(
            "ageing_SXlocation_ruralXage_5_to_SXlocation_ruralXage_15",
            C("S", {"location": "rural", "age": "5"}),
            C("S", {"location": "rural", "age": "15"}),
            10,
        ),
        SojournFlow(
            "ageing_IXlocation_urbanXage_5_to_IXlocation_urbanXage_15",
            C("I", {"location": "urban", "age": "5"}),
            C("I", {"location": "urban", "age": "15"}),
            10,
        ),
        SojournFlow(
            "ageing_IXlocation_ruralXage_5_to_IXlocation_ruralXage_15",
            C("I", {"location": "rural", "age": "5"}),
            C("I", {"location": "rural", "age": "15"}),
            10,
        ),
        SojournFlow(
            "ageing_RXlocation_urbanXage_5_to_RXlocation_urbanXage_15",
            C("R", {"location": "urban", "age": "5"}),
            C("R", {"location": "urban", "age": "15"}),
            10,
        ),
        SojournFlow(
            "ageing_RXlocation_ruralXage_5_to_RXlocation_ruralXage_15",
            C("R", {"location": "rural", "age": "5"}),
            C("R", {"location": "rural", "age": "15"}),
            10,
        ),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_exit_flows():
    """
    Ensure exit flows are stratified correctly.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_death_flow("d_S", 3, "S")
    model.add_death_flow("d_I", 5, "I")
    model.add_death_flow("d_R", 7, "R")

    expected_flows = [
        DeathFlow("d_S", C("S"), 3),
        DeathFlow("d_I", C("I"), 5),
        DeathFlow("d_R", C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply partial stratification
    strat = Stratification("location", ["urban", "rural"], ["S", "I"])
    model.stratify_with(strat)

    expected_flows = [
        DeathFlow("d_S", C("S", {"location": "urban"}), 3),
        DeathFlow("d_S", C("S", {"location": "rural"}), 3),
        DeathFlow("d_I", C("I", {"location": "urban"}), 5),
        DeathFlow("d_I", C("I", {"location": "rural"}), 5),
        DeathFlow("d_R", C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply partial stratification with flow adjustments
    strat = Stratification("age", ["young", "old"], ["I", "R"])
    strat.add_flow_adjustments("d_I", {"young": Multiply(0.5), "old": Multiply(2)})
    strat.add_flow_adjustments("d_R", {"young": Multiply(0.5), "old": Multiply(2)})
    model.stratify_with(strat)

    expected_flows = [
        DeathFlow("d_S", C("S", {"location": "urban"}), 3),
        DeathFlow("d_S", C("S", {"location": "rural"}), 3),
        DeathFlow("d_I", C("I", {"location": "urban", "age": "young"}), 5, [Multiply(0.5)]),
        DeathFlow("d_I", C("I", {"location": "urban", "age": "old"}), 5, [Multiply(2)]),
        DeathFlow("d_I", C("I", {"location": "rural", "age": "young"}), 5, [Multiply(0.5)]),
        DeathFlow("d_I", C("I", {"location": "rural", "age": "old"}), 5, [Multiply(2)]),
        DeathFlow("d_R", C("R", {"age": "young"}), 7, [Multiply(0.5)]),
        DeathFlow("d_R", C("R", {"age": "old"}), 7, [Multiply(2)]),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_add_exit_flows_post_stratification():
    """
    Ensure user can add exit flows post stratification.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert len(model._flows) == 0

    # Apply partial stratification
    strat = Stratification("location", ["urban", "rural"], ["S", "I"])
    model.stratify_with(strat)
    assert len(model._flows) == 0

    model.add_death_flow("d_S", 3, "S")
    model.add_death_flow("d_I", 5, "I")
    model.add_death_flow("d_R", 7, "R")

    expected_flows = [
        DeathFlow("d_S", C("S", {"location": "urban"}), 3),
        DeathFlow("d_S", C("S", {"location": "rural"}), 3),
        DeathFlow("d_I", C("I", {"location": "urban"}), 5),
        DeathFlow("d_I", C("I", {"location": "rural"}), 5),
        DeathFlow("d_R", C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_add_exit_flows_post_stratification__with_filter():
    """
    Ensure user can add exit flows post stratification when a strata filter is applied
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    assert len(model._flows) == 0

    # Apply partial stratification
    strat = Stratification("location", ["urban", "rural"], ["S", "I"])
    model.stratify_with(strat)
    assert len(model._flows) == 0

    model.add_death_flow("d_S", 3, "S", source_strata={"location": "rural"}, expected_flow_count=1)
    model.add_death_flow("d_I", 5, "I", source_strata={"location": "rural"}, expected_flow_count=1)
    model.add_death_flow("d_R", 7, "R", source_strata={"location": "rural"}, expected_flow_count=0)

    expected_flows = [
        DeathFlow("d_S", C("S", {"location": "rural"}), 3),
        DeathFlow("d_I", C("I", {"location": "rural"}), 5),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_source_and_dest_stratified():
    """
    Ensure transition flows are stratified correctly when both the flow source and dest are stratified.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_infection_frequency_flow("infection", 0.03, "S", "I")
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        InfectionFrequencyFlow(
            "infection", C("S"), C("I"), 0.03, model._get_infection_frequency_multiplier
        ),
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = Stratification("location", ["urban", "rural"], ["S", "I", "R"])
    model.stratify_with(strat)

    expected_flows = [
        InfectionFrequencyFlow(
            "infection",
            C("S", {"location": "urban"}),
            C("I", {"location": "urban"}),
            0.03,
            model._get_infection_frequency_multiplier,
        ),
        InfectionFrequencyFlow(
            "infection",
            C("S", {"location": "rural"}),
            C("I", {"location": "rural"}),
            0.03,
            model._get_infection_frequency_multiplier,
        ),
        SojournFlow("recovery", C("I", {"location": "urban"}), C("R", {"location": "urban"}), 7),
        SojournFlow("recovery", C("I", {"location": "rural"}), C("R", {"location": "rural"}), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_source_only_stratified():
    """
    Ensure transition flows are stratified correctly when only the flow source is stratified.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = Stratification("location", ["urban", "rural"], ["S", "I"])
    model.stratify_with(strat)

    expected_flows = [
        SojournFlow("recovery", C("I", {"location": "urban"}), C("R"), 7),
        SojournFlow("recovery", C("I", {"location": "rural"}), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_dest_only_stratified():
    """
    Ensure transition flows are stratified correctly when only the flow destination is stratified.
    Expect an person-conserving adjustment of 1/N to be applied to each flow - N being the number of new strata.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = Stratification("location", ["urban", "rural"], ["R"])
    model.stratify_with(strat)

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R", {"location": "urban"}), 7, [Multiply(0.5)]),
        SojournFlow("recovery", C("I"), C("R", {"location": "rural"}), 7, [Multiply(0.5)]),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_dest_only_stratified__with_adjustments():
    """
    Ensure transition flows are stratified correctly when only the flow destination is stratified.
    Expect adjustments to override the automatic person-conserving adjustment.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = Stratification("location", ["urban", "rural"], ["R"])
    strat.add_flow_adjustments("recovery", {"urban": Overwrite(0.7), "rural": Overwrite(0.1)})
    model.stratify_with(strat)

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R", {"location": "urban"}), 7, [Overwrite(0.7)]),
        SojournFlow("recovery", C("I"), C("R", {"location": "rural"}), 7, [Overwrite(0.1)]),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_dest_only_stratified__with_strains():
    """
    Ensure transition flows are stratified correctly when only the flow destination is stratified.
    Expect the strain stratification to ignore the automatic person-conserving adjustment.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = StrainStratification("location", ["urban", "rural"], ["R"])
    model.stratify_with(strat)

    # No adjustments added
    expected_flows = [
        SojournFlow("recovery", C("I"), C("R", {"location": "urban"}), 7),
        SojournFlow("recovery", C("I"), C("R", {"location": "rural"}), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])


def test_stratify_transition_flows__with_dest_only_stratified__with_adjustments_and_strains():
    """
    Ensure transition flows are stratified correctly when only the flow destination is stratified.
    Expect adjustments to override the automatic person-conserving adjustment when using a strain strat.
    """
    model = CompartmentalModel(
        times=[0, 5], compartments=["S", "I", "R"], infectious_compartments=["I"]
    )
    model.add_sojourn_flow("recovery", 7, "I", "R")

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R"), 7),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])

    # Apply stratification
    strat = StrainStratification("location", ["urban", "rural"], ["R"])
    strat.add_flow_adjustments("recovery", {"urban": Overwrite(0.7), "rural": Overwrite(0.1)})
    model.stratify_with(strat)

    expected_flows = [
        SojournFlow("recovery", C("I"), C("R", {"location": "urban"}), 7, [Overwrite(0.7)]),
        SojournFlow("recovery", C("I"), C("R", {"location": "rural"}), 7, [Overwrite(0.1)]),
    ]
    assert len(expected_flows) == len(model._flows)
    assert all([a._is_equal(e) for e, a in zip(expected_flows, model._flows)])