
from summer2.experimental.model_builder import ParamStruct
from pydantic.dataclasses import dataclass

class ParamConfig:
    """
    Config for parameter models.
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable

@dataclass(config=ParamConfig)
class Parameters(ParamStruct):
    # Metadata
    description: str
    time: dict
    contact_rate: float
    recovery_rate: float
    pop_size: float
    infection_seed: float