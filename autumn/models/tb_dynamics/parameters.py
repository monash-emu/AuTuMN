"""
Type definition for model parameters
"""
from pydantic import BaseModel, Extra
from pydantic.dataclasses import dataclass
from typing import Optional

# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid

"""
Commonly used checking processes
"""


class Time(BaseModel):
    """
    Parameters to define the model time period and evaluation steps.
    """

    start: float
    end: float
    step: float


class ParamConfig:
    """
    Config for parameters model
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    iso3: str
    # Country info
    crude_birth_rate: float
    start_population_size: float
    # Running time
    time: Time
    # Output requests
    infectious_seed: float
