"""
Type definition for model parameters
"""
from typing import List, Optional

from pydantic import BaseModel, Extra, validator
from pydantic.dataclasses import dataclass


# Forbid additional arguments to prevent extraneous parameter specification
BaseModel.Config.extra = Extra.forbid


class Time(BaseModel):
    start: float
    end: float
    step: float
    critical_ranges: List[List[float]]


class ParamConfig:
    """Config for parameter models"""

    anystr_strip_whitespace = True  # Strip whitespace
    allow_mutation = False  # Params should be immutable


@dataclass(config=ParamConfig)
class Parameters:
    # Metadata
    description: Optional[str]
    # Country info
    crude_birth_rate: float
    iso3: str
    region: Optional[str]
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
    pt_destination_compartment: str
    hh_contacts_pt: dict
    awareness_raising: Optional[dict]
    # Other
    inflate_reactivation_for_diabetes: bool
    extra_params: dict
    haario_scaling_factor: float
    metropolis_initialisation: str

    @validator("time_variant_tsr", pre=True, allow_reuse=True)
    def check_time_variant_tsr(val):
        msg = "Treatment success rate should always be > 0."
        assert all([v > 0.0 for v in val.values()]), msg
        return val
