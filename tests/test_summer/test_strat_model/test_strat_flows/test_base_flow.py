from unittest import mock

from summer.flow.base import BaseTransitionFlow, BaseEntryFlow, BaseExitFlow
from summer.compartment import Compartment
from summer.constants import FlowAdjustment

source = Compartment("source")
dest = Compartment("dest")


def test_update_compartment_indices():
    trans_flow = TransitionFlow(source, dest, None, None, None)
    entry_flow = EntryFlow(dest, None, None, None)
    exit_flow = ExitFlow(source, None, None, None)
    mapping = {"source": 2, "dest": 7}
    trans_flow.update_compartment_indices(mapping)
    entry_flow.update_compartment_indices(mapping)
    exit_flow.update_compartment_indices(mapping)
    assert trans_flow.source.idx == 2
    assert exit_flow.source.idx == 2
    assert trans_flow.dest.idx == 7
    assert entry_flow.dest.idx == 7


def test_get_weight_value__with_no_adjustments():
    adjustments = []
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3
    mock_func.assert_called_once_with("foo", 3)


def test_get_weight_value__with_multiply_adjustment():
    adjustments = [(FlowAdjustment.MULTIPLY, 7)]
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3 * 7
    mock_func.assert_called_once_with("foo", 3)


def test_get_weight_value__with_overwrite_adjustment():
    adjustments = [(FlowAdjustment.OVERWRITE, 23)]
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 23
    mock_func.assert_called_once_with("foo", 3)


def test_get_weight_value__with_compose_adjustment():
    adjustments = [(FlowAdjustment.COMPOSE, "bar")]
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3 * 3 * 13
    mock_func.assert_has_calls(
        [mock.call("foo", 3), mock.call("bar", 3),]
    )


def test_get_weight_value__with_many_adjustments():
    adjustments = [
        (FlowAdjustment.MULTIPLY, 17),
        (FlowAdjustment.COMPOSE, "bar"),
    ]
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 2 * 3 * 3 * 13 * 17
    mock_func.assert_has_calls(
        [mock.call("foo", 3), mock.call("bar", 3),]
    )


def test_get_weight_value__with_many_adjustments__and_overwrite():
    adjustments = [
        (FlowAdjustment.MULTIPLY, 17),
        (FlowAdjustment.OVERWRITE, 59),
        (FlowAdjustment.OVERWRITE, 23),
        (FlowAdjustment.COMPOSE, "bar"),
    ]
    mock_func = mock.Mock()
    mock_func.side_effect = _get_param_value
    flow = TransitionFlow(source, dest, "foo", mock_func, adjustments)
    weight = flow.get_weight_value(3)
    assert weight == 23 * 3 * 13
    mock_func.assert_has_calls(
        [mock.call("foo", 3), mock.call("bar", 3),]
    )


def _get_param_value(name, time):
    if name == "foo":
        return 2 * time
    elif name == "bar":
        return 13 * time
    else:
        raise ValueError("broken")


class FlowMixin:
    type = "test"

    def get_net_flow(self, compartment_values, time):
        return 0

    def copy(self, **kwargs):
        return None

    def __repr__(self):
        return ""


class EntryFlow(FlowMixin, BaseEntryFlow):
    def __init__(
        self, dest, param_name, param_func, adjustments,
    ):
        self.adjustments = adjustments
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func


class ExitFlow(FlowMixin, BaseExitFlow):
    def __init__(
        self, source, param_name, param_func, adjustments,
    ):
        self.adjustments = adjustments
        self.source = source
        self.param_name = param_name
        self.param_func = param_func


class TransitionFlow(FlowMixin, BaseTransitionFlow):
    def __init__(
        self, source, dest, param_name, param_func, adjustments,
    ):
        self.adjustments = adjustments
        self.source = source
        self.dest = dest
        self.param_name = param_name
        self.param_func = param_func
