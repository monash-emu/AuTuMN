from typing import List
import pandas as pd
import numpy as np
from summer2 import Stratification, Overwrite, Multiply
from summer2.parameters import Time, Function
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from autumn.model_features.curve import scale_up_function
from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.utils.utils import change_parameter_unit
from autumn.models.tb_dynamics.utils import (
    get_parameter_dict_from_function,
    create_step_function_from_dict,
)
from autumn.models.tb_dynamics2.constants import Compartment, INFECTIOUS_COMPS
from math import log, exp



def get_age_strat(
    params: Parameters,
    compartments: List[str],
    age_pops: pd.Series = None,
    age_mixing_matrix=None,
) -> Stratification:

    """
     Function to create the age group stratification object..

    Args:
        params: Parameter class
        age_pops: The population distribution by age
        age_mixing_matrix: The age-specific mixing matrix
        compartments: All the model compartments

    Returns:
        The age stratification summer object
    """
    age_breakpoints = params.age_breakpoints
    iso3 = params.country.iso3
    strat = Stratification("age", age_breakpoints, compartments)
    # set age mixing matrix
    if age_mixing_matrix is not None:
        strat.set_mixing_matrix(age_mixing_matrix)
    # set age prop split
    if age_pops is not None:
        age_split_props = age_pops / age_pops.sum()
        strat.set_population_split(age_split_props.to_dict())

    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(age_breakpoints, iso3)
    universal_death_funcs = {}
    for age in age_breakpoints:
        universal_death_funcs[age] = Function(
            build_static_sigmoidal_multicurve(death_rate_years, death_rates_by_age[age]), [Time]
        )

    death_adjs = {str(k): Overwrite(v) for k, v in universal_death_funcs.items()}

    strat.set_flow_adjustments("universal_death", death_adjs)

    # Set age-specific latency parameters (early/late activation + stabilisation).
    for flow_name, latency_params in params.age_stratification.items():
        is_activation_flow = flow_name in ["early_activation", "late_activation"]
        if is_activation_flow:
            # Apply progression multiplier.
            latency_params = {
                k: v * params.progression_multiplier for k, v in latency_params.items()
            }
        latency_params = {
            k: v * 365.251 for k, v in latency_params.items()
        }
        adjs = {str(k): Overwrite(v) for k, v in latency_params.items()}
        strat.set_flow_adjustments(flow_name, adjs)

    for comp in INFECTIOUS_COMPS:
        # We assume that infectiousness increases with age
        # A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase  with  age.
        # This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
        inf_adjs = {}
        for i, age_low in enumerate(params.age_breakpoints):
            if i < len(params.age_breakpoints) - 1:
                age_up = params.age_breakpoints[i + 1]
                # Calculate the average of the sigmoidal function(x -> 1 / (1 + exp(-(x-alpha)))) between the age bounds
                average_infectiousness = (
                    log(1 + exp(age_up - params.age_infectiousness_switch))
                    - log(1 + exp(age_low - params.age_infectiousness_switch))
                ) / (age_up - age_low)
            else:
                # Set infectiousness to 1. for the oldest age group
                average_infectiousness = 1.0

            if comp == Compartment.ON_TREATMENT:
                # Apply infectiousness multiplier for people on treatment,
                average_infectiousness *= params.on_treatment_infect_multiplier

            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)
    # Set age-specific treatment recovery, relapse and treatment death rates
    tsr_curve_func = build_static_sigmoidal_multicurve(
        list(params.time_variant_tsr.keys()), list(params.time_variant_tsr.values()))
    time_variant_tsr = Function(tsr_curve_func, [Time])
    treatment_recovery_funcs = {}
    def get_treatment_recovery_rate(t, treatment_duration, prop_death, death_rate, tsr):
        floor_val = 1 / treatment_duration
        dynamic_val = (
            death_rate
                / prop_death
                * (1.0 / (1.0 - tsr) - 1.0)
        )
        return(max(floor_val, dynamic_val))

    for age in params.age_breakpoints:
        death_rate = universal_death_funcs[age]
        treatment_recovery_funcs[age] = Function(get_treatment_recovery_rate, [Time, params.treatment_duration, params.prop_death_among_negative_tx_outcome, death_rate,  time_variant_tsr])
           
    treatment_recovery_adjs = {str(k): Multiply(v) for k, v in treatment_recovery_funcs.items()}
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_adjs)
        # def make_get_treatment_recovery_rate(t, age):
        #     return Function(get_treatment_recovery_rate, [Time, age])
        # treatment_recovery_funcs[age] = make_get_treatment_recovery_rate
      

   

    return strat
