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

class MixingMatrices(BaseModel):
    type: Optional[str]  # None defaults to Prem matrices, otherwise 'prem' or 'extrapolated' - see build_model
    source_iso3: Optional[str]
    age_adjust: Optional[bool]  # Only relevant if 'extrapolated' selected


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    iso3: str
    age_mixing: Optional[MixingMatrices]
    # Country info
    start_population_size: float
    crude_birth_rate: float
    crude_death_rate: float
    # Running time
    time: Time
    # Model structure
    age_breakpoints: list
    infectious_seed: float
    cumulative_start_time: float
    # Base TB model
    contact_rate: float
    rr_infection_latent: float
    rr_infection_recovered: float
    age_specific_latency: dict
    progression_multiplier: float
    self_recovery_rate: float
    infect_death_rate: float
