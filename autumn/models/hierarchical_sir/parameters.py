"""
Type definition for model parameters
"""
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

from datetime import date
from typing import Any, Dict, List, Optional, Union

from autumn.models.covid_19.constants import GOOGLE_MOBILITY_LOCATIONS
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.inputs.social_mixing.constants import LOCATIONS

BASE_DATE = COVID_BASE_DATETIME.date()

# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid

class Time(BaseModel):
    """
    Parameters to define the model time period and evaluation steps.
    """

    start: float
    end: float
    step: float

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        start, end = values.get("start"), values.get("end")
        assert end >= start, f"End time: {end} before start: {start}"
        return values


class ParamConfig:
    """
    Config for parameter models
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata    
    time: Time
    description: str
    # Values
    hyper_beta: float
    beta: dict
    
    gamma: float
    infectious_seed: float    
    total_pop: float
    location_split: dict