"""
Type definition for model parameters
"""
from numpy import int_
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Union
from autumn.core.inputs.social_mixing.constants import LOCATIONS

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
    critical_range: List[List[float]]

    @root_validator(pre=True, allow_reuse=True)
    def check_lengths(cls, values):
        start, end = values.get("start"), values.get("end")
        assert end >= start, f"End time: {end} before start: {start}"
        return values


class ParamConfig:
    """
    Config for parameters model
    """

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


class MixingMatrices(BaseModel):
    """
    Config mixing matrices. None defaults and "prem" to Prem matrices. "extrapolated" for building synthetic matrices with age adjustment (age_adjust  = True)
    """

    type: Optional[str]
    source_iso3: Optional[str]
    age_adjust: Optional[bool]


class Population(BaseModel):
    """
    Model population parameters.
    """

    region: Optional[str]  # None/null means default to parent country
    year: int  # Year to use to find the population data in the database

    @validator("year", pre=True, allow_reuse=True)
    def check_year(year):
        msg = f"Year before 1800 or after 2050: {year}"
        assert 1800 <= year <= 2050, msg
        return year


class CompartmentSojourn(BaseModel):
    """
    Compartment sojourn times, meaning the mean period of time spent in a compartment.
    """

    total_time: float
    proportion_early: Optional[float]


class Sojourns(BaseModel):
    """
    Parameters for determining how long a person stays in a given compartment.
    """

    active: CompartmentSojourn
    latent: CompartmentSojourn
    recovered: Optional[float]  # Doesn't have an early and late


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    iso3: str
    # Country info
    population: Population
    crude_birth_rate: float
    age_mixing: Optional[MixingMatrices]
    # Running time.
    time: Time
    # Output requests
