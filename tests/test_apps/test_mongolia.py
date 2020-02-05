from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.mongolia.mongolia_tb_model import build_mongolia_model, run_model


def test_build_mongolia_model():
    """
    Ensure we can build the Mongolia model with nothing crashing.
    """
    model = build_mongolia_model({})
    assert type(model) is StratifiedModel


@mock.patch("applications.mongolia.mongolia_tb_model.STRATIFY_BY", ["age"])
@mock.patch("autumn.tb_model.outputs.Outputs")
def test_run_mongolia_model(mock_output_cls):
    """
    Ensure Mongolia model runs with nothing crashing.
    """
    run_model()
