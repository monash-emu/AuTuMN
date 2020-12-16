from summer2 import Compartment, adjust
from summer2.flows import BaseTransitionFlow, BaseEntryFlow, BaseExitFlow


class TransitionFlow(BaseTransitionFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


class EntryFlow(BaseEntryFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


class ExitFlow(BaseExitFlow):
    def get_net_flow(self, compartment_values, time):
        return 1


def test_is_match():
    """
    Ensure is_match is behaving sensibly.
    TODO: Come up with a more rigorous, parametrized check.
    """
    source = Compartment("S")
    dest = Compartment("I")

    trans_flow = TransitionFlow("x", source, dest, None, None)
    assert trans_flow.is_match("x", source_strata={}, dest_strata={})
    assert not trans_flow.is_match("y", source_strata={}, dest_strata={})
    assert not trans_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert not trans_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})

    entry_flow = EntryFlow("x", dest, None, None)
    assert entry_flow.is_match("x", source_strata={}, dest_strata={})
    assert not entry_flow.is_match("y", source_strata={}, dest_strata={})
    assert entry_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert not entry_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})

    exit_flow = ExitFlow("x", source, None, None)
    assert exit_flow.is_match("x", source_strata={}, dest_strata={})
    assert not exit_flow.is_match("y", source_strata={}, dest_strata={})
    assert not exit_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert exit_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})

    source = Compartment("S", {"thing": "good", "color": "red"})
    dest = Compartment("I", {"thing": "good", "color": "red"})

    trans_flow = TransitionFlow("x", source, dest, None, None)
    assert trans_flow.is_match("x", source_strata={}, dest_strata={})
    assert not trans_flow.is_match("y", source_strata={}, dest_strata={})

    assert trans_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert trans_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})
    assert trans_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={"color": "red"})
    assert trans_flow.is_match("x", source_strata={"thing": "good", "color": "red"}, dest_strata={})
    assert trans_flow.is_match("x", source_strata={}, dest_strata={"thing": "good", "color": "red"})
    assert not trans_flow.is_match(
        "x", source_strata={"thing": "good"}, dest_strata={"color": "green"}
    )
    assert not trans_flow.is_match("x", source_strata={"thing": "bad"}, dest_strata={})
    assert not trans_flow.is_match("x", source_strata={}, dest_strata={"thing": "bad"})

    entry_flow = EntryFlow("x", dest, None, None)
    assert entry_flow.is_match("x", source_strata={}, dest_strata={})
    assert not entry_flow.is_match("y", source_strata={}, dest_strata={})

    assert entry_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert entry_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})
    assert entry_flow.is_match("x", source_strata={"thing": "bad"}, dest_strata={})
    assert not entry_flow.is_match("x", source_strata={}, dest_strata={"thing": "bad"})

    exit_flow = ExitFlow("x", source, None, None)
    assert exit_flow.is_match("x", source_strata={}, dest_strata={})
    assert not exit_flow.is_match("y", source_strata={}, dest_strata={})
    assert exit_flow.is_match("x", source_strata={"thing": "good"}, dest_strata={})
    assert exit_flow.is_match("x", source_strata={}, dest_strata={"thing": "good"})
    assert not exit_flow.is_match("x", source_strata={"thing": "bad"}, dest_strata={})
    assert exit_flow.is_match("x", source_strata={}, dest_strata={"thing": "bad"})


SOURCE = Compartment("source")
DEST = Compartment("dest")


def test_update_compartment_indices():
    mapping = {"source": 2, "dest": 7}

    trans_flow = TransitionFlow("x", SOURCE, DEST, None, None)
    trans_flow.update_compartment_indices(mapping)
    assert trans_flow.source.idx == 2
    assert trans_flow.dest.idx == 7

    entry_flow = EntryFlow("x", DEST, None, None)
    entry_flow.update_compartment_indices(mapping)
    assert entry_flow.dest.idx == 7

    exit_flow = ExitFlow("x", SOURCE, None, None)
    exit_flow.update_compartment_indices(mapping)
    assert exit_flow.source.idx == 2


def test_get_weight_value__with_no_adjustments():
    flow = TransitionFlow(
        name="recovery", source=SOURCE, dest=DEST, param=lambda t: 2 * t, adjustments=[]
    )
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3


def test_get_weight_value__with_multiply_adjustments():
    flow = TransitionFlow(
        name="recovery",
        source=SOURCE,
        dest=DEST,
        param=lambda t: 2 * t,
        adjustments=[adjust.Multiply(5), adjust.Multiply(lambda t: 7)],
    )
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3 * 5 * 7


def test_get_weight_value__with_overwrite_adjustment():
    flow = TransitionFlow(
        name="recovery",
        source=SOURCE,
        dest=DEST,
        param=lambda t: 2 * t,
        adjustments=[
            adjust.Multiply(2),
            adjust.Overwrite(5),  # Overwrites 2 * 3, which is ignored.
            adjust.Multiply(7),
        ],
    )
    weight = flow.get_weight_value(3)
    assert weight == 5 * 7


def test_get_weight_value__with_muiltiple_overwrite_adjustments():
    flow = TransitionFlow(
        name="recovery",
        source=SOURCE,
        dest=DEST,
        param=lambda t: 2 * t,
        adjustments=[
            adjust.Overwrite(5),
            adjust.Multiply(7),
            adjust.Overwrite(lambda t: 13),  # Last overwrite adjustment wins
        ],
    )
    weight = flow.get_weight_value(3)
    assert weight == 13
