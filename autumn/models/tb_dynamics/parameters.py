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
    stratify_by: list
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
    time_variant_tsr: dict
    infect_death_rate: float
    time_variant_bcg_perc: dict
    bcg_effect: str

    # Characterising age stratification
    age_infectiousness_switch: float

    # Treatment
    treatment_duration: float
    time_variant_tsr: dict
    prop_death_among_negative_tx_outcome: float
    on_treatment_infect_multiplier: float

    # Defining sites of infection
    incidence_props_pulmonary: float
    incidence_props_smear_positive_among_pulmonary: float
    smear_negative_infect_multiplier: float
    extrapulmonary_infect_multiplier: float

    #intervention
    awareness_raising: Optional[dict]

    # Detection
    time_variant_tb_screening_rate: dict
    passive_screening_sensitivity: dict

    self_recovery_rate_dict: dict
    infect_death_rate_dict: dict

