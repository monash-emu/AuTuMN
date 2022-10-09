from typing import List
import pandas as pd
import numpy as np
from summer import AgeStratification, Overwrite, Multiply
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve import scale_up_function
from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.utils.utils import change_parameter_unit
from autumn.models.tb_dynamics.utils import (
    get_parameter_dict_from_function,
    create_step_function_from_dict,
)

def get_age_strat( 
    params: Parameters,
    compartments: List[str],
    age_pops: pd.Series = None,
    age_mixing_matrix = None,
) -> AgeStratification:

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
    iso3 = params.iso3
    strat = AgeStratification("age", age_breakpoints, compartments)
    # set age mixing matrix
    if age_mixing_matrix is not None:
        strat.set_mixing_matrix(age_mixing_matrix)
    #set age prop split
    if age_pops is not None:    
        age_split_props = age_pops / age_pops.sum()
        strat.set_population_split(age_split_props.to_dict())

    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(age_breakpoints, iso3)
    universal_death_funcs = {}
    for age in age_breakpoints:
        universal_death_funcs[age] = scale_up_function(
            death_rate_years, death_rates_by_age[age], smoothness=0.2, method=5
        )

    death_adjs = {str(k): Overwrite(v) for k, v in universal_death_funcs.items()}
    for comp in compartments:
        flow_name = f"universal_death_for_{comp}"
        strat.set_flow_adjustments(flow_name, death_adjs)

     # Set age-specific latency parameters (early/late activation + stabilisation).
    for flow_name, latency_params in params.age_specific_latency.items():
        is_activation_flow = flow_name in ["early_activation", "late_activation"]
        if is_activation_flow:
            # Apply progression multiplier.
            latency_params = {
                k: v * params.progression_multiplier for k, v in latency_params.items()
            }

        step_func = create_step_function_from_dict(latency_params)
        step_func_as_dict = get_parameter_dict_from_function(step_func, params.age_breakpoints)
        adjs = change_parameter_unit(step_func_as_dict, 365.251)
        adjs = {str(k): Multiply(v) for k, v in adjs.items()}
        strat.set_flow_adjustments(flow_name, adjs)

    return strat
