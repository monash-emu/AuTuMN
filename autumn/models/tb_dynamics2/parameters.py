"""
Type definition for model parameters
"""
from pydantic import BaseModel as _BaseModel, Extra, validator
from pydantic.dataclasses import dataclass
from typing import Optional, Dict

from summer2.experimental.model_builder import (
    ParamStruct,
    parameter_class as pclass
    )

# Forbid additional arguments to prevent extraneous parameter specification
_BaseModel.Config.extra = Extra.forbid

"""
Commonly used checking processes
"""

class BaseModel(_BaseModel, ParamStruct):
    pass

class Time(BaseModel):
    """
    Parameters to define the model time period and evaluation steps.
    """

    start: float
    end: float
    step: float

class Country(BaseModel):
    """
    The country that the model is based in. (The country may be, and often is, the same as the region.)
    """

    iso3: str
    country_name: str

    @validator("iso3", pre=True, allow_reuse=True)
    def check_length(iso3):
        assert len(iso3) == 3, f"ISO3 codes should have three digits, code is: {iso3}"
        return iso3
    

class StatifiedRate(BaseModel):
    unstratified: pclass()
    smear_positive: pclass()
    smear_negative: pclass()

class Props(BaseModel):
    male: float
    female: float

class Multipliers(BaseModel):
    male: pclass()
    female: pclass()


class Adjustments(BaseModel):
    infection: Multipliers
    detection: Multipliers

class GenderStrat(BaseModel):
    strata: list
    proportions: dict
    adjustments: Adjustments




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
class Parameters(ParamStruct):
    # Metadata
    description: Optional[str]
    country: Country
    age_mixing: Optional[MixingMatrices]
    # Country info
    start_population_size: pclass()
    crude_birth_rate: float
    crude_death_rate: float
    # Running time
    time: Time
    # Model structure
    stratify_by: list
    age_breakpoints: Optional[list]
    infectious_seed: float
    cumulative_start_time: float
    # Base TB model
    contact_rate: pclass()
    rr_infection_latent: pclass()
    rr_infection_recovered: pclass()
    progression_multiplier: pclass()
    infect_death_rate_dict: StatifiedRate
    self_recovery_rate_dict: StatifiedRate
    
    # Characterising age stratification
    age_infectiousness_switch: float
    age_stratification: dict

    calculated_outputs: list
    outputs_stratification: dict
    cumulative_output_start_time: float
    
      # Detection
    cdr_adjustment: pclass()
    time_variant_tb_screening_rate: dict
    passive_screening_sensitivity: dict
    time_variant_tsr: dict

    treatment_duration: float
    prop_death_among_negative_tx_outcome: float
    on_treatment_infect_multiplier: float

    time_variant_bcg_perc: dict
    bcg_effect: str
    incidence_props_pulmonary: float
    incidence_props_smear_positive_among_pulmonary: float
    smear_negative_infect_multiplier: float
    extrapulmonary_infect_multiplier: float
    gender: GenderStrat




