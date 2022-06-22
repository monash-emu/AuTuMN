"""
Type definition for model parameters
"""
from datetime import date
from typing import Any, Dict, List, Optional, Union

# from autumn.settings.constants import COVID_BASE_DATETIME, GOOGLE_MOBILITY_LOCATIONS, COVID_BASE_AGEGROUPS
from autumn.core.inputs.social_mixing.constants import LOCATIONS
from numpy import int_
from pydantic import BaseModel, Extra, root_validator, validator
from pydantic.dataclasses import dataclass

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


class Country(BaseModel):
    """
    The country that the model is based in. (The country may be, and often is, the same as the region.)
    """

    iso3: str

    @validator("iso3", pre=True, allow_reuse=True)
    def check_length(iso3):
        assert len(iso3) == 3, f"ISO3 codes should have three digits, code is: {iso3}"
        return iso3


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
    country: Country
    # Country info
    population: Population
    crude_birth_rate: float
    age_mixing: Optional[MixingMatrices]
    # Running time.
    time: Time
    # Output requests
    calculated_outputs: list
    outputs_stratification: dict
    cumulative_output_start_time: float
    # Model structure
    stratify_by: list
    age_breakpoints: list
    user_defined_stratifications: dict
    # Demographics
    start_population_size: float
    universal_death_rate: float
    # Base disease model
    contact_rate: float
    age_specific_latency: dict
    progression_multiplier: float
    self_recovery_rate_dict: dict
    infect_death_rate_dict: dict
    rr_infection_latent: float
    rr_infection_recovered: float
    time_variant_bcg_perc: dict
    bcg_effect: str
    import_ltbi_cases: Optional[dict]
    # Detection
    time_variant_tb_screening_rate: dict
    passive_screening_sensitivity: dict
    # Treatment
    treatment_duration: float
    time_variant_tsr: dict
    prop_death_among_negative_tx_outcome: float
    on_treatment_infect_multiplier: float
    # Characterising age stratification
    age_infectiousness_switch: float
    # Defining organ stratification
    incidence_props_pulmonary: float
    incidence_props_smear_positive_among_pulmonary: float
    smear_negative_infect_multiplier: float
    extrapulmonary_infect_multiplier: float
    # Interventions
    time_variant_acf: list
    acf_screening_sensitivity: float
    time_variant_ltbi_screening: list
    ltbi_screening_sensitivity: float
    pt_efficacy: float
    pt_sae_prop: float
    pt_destination_compartment: str
    hh_contacts_pt: dict
    awareness_raising: Optional[dict]
    # Other
    inflate_reactivation_for_diabetes: bool
    extra_params: dict
    haario_scaling_factor: float
    metropolis_initialisation: str
