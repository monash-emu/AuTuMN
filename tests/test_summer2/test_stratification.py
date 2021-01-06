"""
Unit tests for the Stratification model.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from summer2 import (
    Stratification,
    AgeStratification,
    StrainStratification,
    Compartment,
    Multiply,
    Overwrite,
)
from summer2.flows import BaseExitFlow, BaseEntryFlow, BaseTransitionFlow


class TransitionFlow(BaseTransitionFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


class EntryFlow(BaseEntryFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


class ExitFlow(BaseExitFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


def test_create_stratification():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.name == "location"
    assert strat.compartments == [Compartment("S"), Compartment("I"), Compartment("R")]
    assert strat.strata == ["rural", "urban"]
    assert strat.population_split == {"rural": 0.5, "urban": 0.5}
    assert strat.flow_adjustments == {}
    assert strat.infectiousness_adjustments == {}
    assert strat.mixing_matrix is None
    assert not strat.is_ageing()
    assert not strat.is_strain()


def test_create_strain_stratification():
    strat = StrainStratification(
        name="strain", strata=["mild", "horrible"], compartments=["S", "I", "R"]
    )
    assert not strat.is_ageing()
    assert strat.is_strain()


def test_create_age_stratification():
    strat = AgeStratification(name="age", strata=["0", "5", "10"], compartments=["S", "I", "R"])
    assert strat.is_ageing()
    assert not strat.is_strain()

    # Fails coz non integer stratum
    with pytest.raises(AssertionError):
        AgeStratification(name="age", strata=["0", "hello", "10"], compartments=["S", "I", "R"])

    # Fails coz no zero age
    with pytest.raises(AssertionError):
        AgeStratification(name="age", strata=["1", "5", "10"], compartments=["S", "I", "R"])


def test_create_stratification__with_pop_split():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.population_split == {"rural": 0.5, "urban": 0.5}
    # Works
    strat.set_population_split({"rural": 0.2, "urban": 0.8})
    assert strat.population_split == {"rural": 0.2, "urban": 0.8}

    # Fails coz missing a key
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": 1})

    # Fails coz doesn't sum to 1
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": 0.2, "rural": 0.3})

    # Fails coz contains negative number
    with pytest.raises(AssertionError):
        strat.set_population_split({"urban": -2, "rural": 3})


def test_create_stratification__with_flow_adjustments():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.flow_adjustments == {}

    # Fail coz not all strata specified
    with pytest.raises(AssertionError):
        strat.add_flow_adjustments(
            flow_name="recovery",
            adjustments={"rural": Multiply(1.2)},
        )

    # Fail coz an incorrect strata specified
    with pytest.raises(AssertionError):
        strat.add_flow_adjustments(
            flow_name="recovery",
            adjustments={
                "rural": Multiply(1.2),
                "urban": Multiply(0.8),
                "alpine": Multiply(1.1),
            },
        )

    strat.add_flow_adjustments(
        flow_name="recovery",
        adjustments={"rural": Multiply(1.2), "urban": Multiply(0.8)},
    )

    assert len(strat.flow_adjustments["recovery"]) == 1
    adj, src, dst = strat.flow_adjustments["recovery"][0]
    assert adj["rural"]._is_equal(Multiply(1.2))
    assert adj["urban"]._is_equal(Multiply(0.8))
    assert not (src or dst)

    # Add another adjustment for the same flow.
    strat.add_flow_adjustments(
        flow_name="recovery",
        adjustments={"rural": Multiply(1.3), "urban": Multiply(0.9)},
        source_strata={"age": "10"},
        dest_strata={"work": "office"},
    )
    assert len(strat.flow_adjustments["recovery"]) == 2
    adj, src, dst = strat.flow_adjustments["recovery"][0]
    assert adj["rural"]._is_equal(Multiply(1.2))
    assert adj["urban"]._is_equal(Multiply(0.8))
    assert not (src or dst)
    adj, src, dst = strat.flow_adjustments["recovery"][1]
    assert adj["rural"]._is_equal(Multiply(1.3))
    assert adj["urban"]._is_equal(Multiply(0.9))
    assert src == {"age": "10"}
    assert dst == {"work": "office"}

    def urban_infection_adjustment(t):
        return 2 * t

    strat.add_flow_adjustments(
        flow_name="infection",
        adjustments={
            "rural": Multiply(urban_infection_adjustment),
            "urban": None,
        },
    )
    assert len(strat.flow_adjustments["infection"]) == 1
    adj, src, dst = strat.flow_adjustments["infection"][0]
    assert adj["rural"]._is_equal(Multiply(urban_infection_adjustment))
    assert adj["urban"] is None
    assert not (src or dst)


def test_get_flow_adjustments__with_no_adjustments():
    trans_flow = TransitionFlow("flow", Compartment("S"), Compartment("I"), 1)
    entry_flow = EntryFlow("flow", Compartment("S"), 1)
    exit_flow = ExitFlow("flow", Compartment("I"), 1)

    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])

    for flow in [trans_flow, entry_flow, exit_flow]:
        assert strat.get_flow_adjustment(flow) is None


def test_get_flow_adjustments__with_one_adjustment():
    other_flow = TransitionFlow("other", Compartment("S"), Compartment("I"), 1)
    trans_flow = TransitionFlow("flow", Compartment("S"), Compartment("I"), 1)
    entry_flow = EntryFlow("flow", Compartment("S"), 1)
    exit_flow = ExitFlow("flow", Compartment("I"), 1)

    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    strat.add_flow_adjustments("flow", {"rural": Multiply(1), "urban": None})

    assert strat.get_flow_adjustment(other_flow) is None
    for flow in [trans_flow, entry_flow, exit_flow]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["urban"] is None
        assert adj["rural"]._is_equal(Multiply(1))


def test_get_flow_adjustments__with_multiple_adjustments():
    other_flow = TransitionFlow("other", Compartment("S"), Compartment("I"), 1)
    trans_flow = TransitionFlow("flow", Compartment("S"), Compartment("I"), 1)
    entry_flow = EntryFlow("flow", Compartment("S"), 1)
    exit_flow = ExitFlow("flow", Compartment("I"), 1)

    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])

    strat.add_flow_adjustments("flow", {"rural": Multiply(1), "urban": None})
    strat.add_flow_adjustments("flow", {"rural": Multiply(3), "urban": Overwrite(2)})

    # Latest flow adjustment should always win.
    assert strat.get_flow_adjustment(other_flow) is None
    for flow in [trans_flow, entry_flow, exit_flow]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(3))
        assert adj["urban"]._is_equal(Overwrite(2))


def test_get_flow_adjustments__with_strata_whitelist():
    # Latest matching flow adjustment should always win.
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    strat.add_flow_adjustments("flow", {"rural": Multiply(1), "urban": None})
    strat.add_flow_adjustments("flow", {"rural": Multiply(3), "urban": Overwrite(2)})
    strat.add_flow_adjustments(
        "flow", {"rural": Multiply(2), "urban": Overwrite(1)}, source_strata={"age": "20"}
    )

    # No source strata
    entry_flow = EntryFlow("flow", Compartment("S"), 1)
    with pytest.raises(AssertionError):
        strat.get_flow_adjustment(entry_flow)

    other_flow = TransitionFlow("other", Compartment("S"), Compartment("I"), 1)
    assert strat.get_flow_adjustment(other_flow) is None

    trans_flow = TransitionFlow("flow", Compartment("S"), Compartment("I"), 1)
    exit_flow = ExitFlow("flow", Compartment("I"), 1)
    for flow in [trans_flow, exit_flow]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(3))
        assert adj["urban"]._is_equal(Overwrite(2))

    # Only flows with matching strata should get the adjustment
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    strat.add_flow_adjustments("flow", {"rural": Multiply(1), "urban": None})
    strat.add_flow_adjustments("flow", {"rural": Multiply(3), "urban": Overwrite(2)})
    strat.add_flow_adjustments(
        "flow", {"rural": Multiply(2), "urban": Overwrite(1)}, dest_strata={"age": "20"}
    )

    # No dest strata
    exit_flow = ExitFlow("flow", Compartment("I"), 1)
    with pytest.raises(AssertionError):
        strat.get_flow_adjustment(exit_flow)

    # No matching dest strata
    other_flow = TransitionFlow("other", Compartment("S"), Compartment("I"), 1)
    other_flow_strat = TransitionFlow("other", Compartment("S"), Compartment("I", {"age": "20"}), 1)
    assert strat.get_flow_adjustment(other_flow) is None
    assert strat.get_flow_adjustment(other_flow_strat) is None

    # Flows without age 20 get the last match.
    trans_flow = TransitionFlow("flow", Compartment("S"), Compartment("I"), 1)
    entry_flow = EntryFlow("flow", Compartment("S"), 1)
    trans_flow_strat_wrong = TransitionFlow(
        "flow", Compartment("S"), Compartment("I", {"age": "10"}), 1
    )
    entry_flow_strat_wrong = EntryFlow("flow", Compartment("S", {"age": "10"}), 1)
    trans_flow_strat_wrong_2 = TransitionFlow(
        "flow", Compartment("S", {"age": "20"}), Compartment("I"), 1
    )
    for flow in [
        trans_flow,
        entry_flow,
        trans_flow_strat_wrong,
        entry_flow_strat_wrong,
        trans_flow_strat_wrong_2,
    ]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(3))
        assert adj["urban"]._is_equal(Overwrite(2))

    trans_flow_strat = TransitionFlow("flow", Compartment("S"), Compartment("I", {"age": "20"}), 1)
    entry_flow_strat = EntryFlow("flow", Compartment("S", {"age": "20"}), 1)
    for flow in [trans_flow_strat, entry_flow_strat]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(2))
        assert adj["urban"]._is_equal(Overwrite(1))

    # The last created matching flow adjustment will win, also include both source and dest.
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    strat.add_flow_adjustments("flow", {"rural": Multiply(1), "urban": None})
    strat.add_flow_adjustments(
        "flow",
        {"rural": Multiply(5), "urban": Overwrite(7)},
        source_strata={"age": "20"},
        dest_strata={"age": "30", "work": "home"},
    )
    strat.add_flow_adjustments(
        "flow",
        {"rural": Multiply(2), "urban": Overwrite(1)},
        source_strata={"age": "20"},
        dest_strata={"age": "30"},
    )
    strat.add_flow_adjustments("flow", {"rural": Multiply(3), "urban": Overwrite(2)})

    # Missing source strata
    trans_flow_strat_wrong = TransitionFlow(
        "flow", Compartment("S"), Compartment("I", {"age": "30", "work": "home"}), 1
    )
    # Missing dest strata
    trans_flow_strat_wrong_2 = TransitionFlow(
        "flow", Compartment("S", {"age": "20"}), Compartment("I"), 1
    )
    # Incomplete dest strata - less specific still wins because of ordering.
    trans_flow_strat_wrong_3 = TransitionFlow(
        "flow", Compartment("S", {"age": "20"}), Compartment("I", {"age": "30"}), 1
    )
    for flow in [trans_flow_strat_wrong, trans_flow_strat_wrong_2, trans_flow_strat_wrong_3]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(3))
        assert adj["urban"]._is_equal(Overwrite(2))

    # Match to the last created stratification
    trans_flow_strat = TransitionFlow(
        "flow", Compartment("S", {"age": "20"}), Compartment("I", {"age": "30", "work": "home"}), 1
    )
    for flow in [trans_flow_strat]:
        adj = strat.get_flow_adjustment(flow)
        assert adj["rural"]._is_equal(Multiply(3))
        assert adj["urban"]._is_equal(Overwrite(2))


def test_create_stratification__with_infectiousness_adjustments():
    strat = Stratification(name="location", strata=["rural", "urban"], compartments=["S", "I", "R"])
    assert strat.infectiousness_adjustments == {}

    # Fail coz not all strata specified
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={"rural": Multiply(1.2)},
        )

    # Fail coz an incorrect strata specified
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": Multiply(1.2),
                "urban": Multiply(0.8),
                "alpine": Multiply(1.1),
            },
        )

    # Fail coz a time-varying function was used (not allowed!)
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": Multiply(1.2),
                "urban": Multiply(lambda t: 2),
            },
        )

    strat.add_infectiousness_adjustments(
        compartment_name="S",
        adjustments={
            "rural": Multiply(1.2),
            "urban": Multiply(2),
        },
    )

    assert strat.infectiousness_adjustments["S"]["rural"]._is_equal(Multiply(1.2))
    assert strat.infectiousness_adjustments["S"]["urban"]._is_equal(Multiply(2))

    # Fail coz we just did this
    with pytest.raises(AssertionError):
        strat.add_infectiousness_adjustments(
            compartment_name="S",
            adjustments={
                "rural": Multiply(1.2),
                "urban": Multiply(2),
            },
        )

    strat.add_infectiousness_adjustments(
        compartment_name="I",
        adjustments={
            "rural": Multiply(1.2),
            "urban": None,
        },
    )

    assert strat.infectiousness_adjustments["I"]["rural"]._is_equal(Multiply(1.2))
    assert strat.infectiousness_adjustments["I"]["urban"] is None


def test_stratify_compartments__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comps = strat._stratify_compartments(comps)
    assert strat_comps == [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I", {"age": "0"}),
        Compartment("I", {"age": "10"}),
        Compartment("I", {"age": "20"}),
        Compartment("R", {"age": "0"}),
        Compartment("R", {"age": "10"}),
        Compartment("R", {"age": "20"}),
    ]


def test_stratify_compartments__with_no_extisting_strat_and_subset_only():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    strat_comps = strat._stratify_compartments(comps)
    assert strat_comps == [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I"),
        Compartment("R"),
    ]


def test_stratify_compartments__with_extisting_strat():
    age_strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    age_comps = age_strat._stratify_compartments(comps)
    loc_strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
    )
    loc_comps = loc_strat._stratify_compartments(age_comps)
    assert loc_comps == [
        Compartment("S", {"age": "0", "location": "rural"}),
        Compartment("S", {"age": "0", "location": "urban"}),
        Compartment("S", {"age": "10", "location": "rural"}),
        Compartment("S", {"age": "10", "location": "urban"}),
        Compartment("S", {"age": "20", "location": "rural"}),
        Compartment("S", {"age": "20", "location": "urban"}),
        Compartment("I", {"age": "0", "location": "rural"}),
        Compartment("I", {"age": "0", "location": "urban"}),
        Compartment("I", {"age": "10", "location": "rural"}),
        Compartment("I", {"age": "10", "location": "urban"}),
        Compartment("I", {"age": "20", "location": "rural"}),
        Compartment("I", {"age": "20", "location": "urban"}),
        Compartment("R", {"age": "0", "location": "rural"}),
        Compartment("R", {"age": "0", "location": "urban"}),
        Compartment("R", {"age": "10", "location": "rural"}),
        Compartment("R", {"age": "10", "location": "urban"}),
        Compartment("R", {"age": "20", "location": "rural"}),
        Compartment("R", {"age": "20", "location": "urban"}),
    ]


def test_stratify_compartment_values__with_no_extisting_strat():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S", "I", "R"],
    )
    strat.set_population_split({"0": 0.25, "10": 0.5, "20": 0.25})
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]

    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
    expected_arr = np.array([250, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_stratify_compartment_values__with_subset_stratified():
    strat = Stratification(
        name="age",
        strata=["0", "10", "20"],
        compartments=["S"],
    )
    strat.set_population_split({"0": 0.25, "10": 0.5, "20": 0.25})
    comps = [Compartment("S"), Compartment("I"), Compartment("R")]
    comp_values = np.array([1000.0, 100.0, 0.0])
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
    expected_arr = np.array([250.0, 500.0, 250.0, 100.0, 0.0])
    assert_array_equal(expected_arr, new_comp_values)


def test_stratify_compartment_values__with_extisting_strat():
    """
    Stratify compartments for the second time, expect that compartments
    are are split according to proportions and old compartments are removed.
    """
    comp_values = np.array([250.0, 500.0, 250.0, 25.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    comps = [
        Compartment("S", {"age": "0"}),
        Compartment("S", {"age": "10"}),
        Compartment("S", {"age": "20"}),
        Compartment("I", {"age": "0"}),
        Compartment("I", {"age": "10"}),
        Compartment("I", {"age": "20"}),
        Compartment("R", {"age": "0"}),
        Compartment("R", {"age": "10"}),
        Compartment("R", {"age": "20"}),
    ]
    strat = Stratification(
        name="location",
        strata=["rural", "urban"],
        compartments=["S", "I", "R"],
    )
    strat.set_population_split({"rural": 0.1, "urban": 0.9})
    new_comp_values = strat._stratify_compartment_values(comps, comp_values)
    expected_arr = np.array(
        [
            25,
            225.0,
            50.0,
            450.0,
            25.0,
            225.0,
            2.5,
            22.5,
            5.0,
            45.0,
            2.5,
            22.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert_array_equal(expected_arr, new_comp_values)
