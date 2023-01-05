from summer2 import CompartmentalModel
from summer2.experimental.model_builder import ModelBuilder


def get_base_params():
    base_params = Params(
        str(Path(__file__).parent.resolve() / "params.yml"),
        validator=lambda d: Parameters(**d),
        validate=False,
    )
    return base_params


def build_model(params: dict, ret_builder=False) -> CompartmentalModel:

    builder = ModelBuilder(params, Parameters)
    params = builder.params



    builder.set_model(model)
    if ret_builder:
        return model, builder
    else:
        return model