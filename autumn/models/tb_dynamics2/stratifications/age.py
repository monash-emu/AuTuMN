from typing import List
import pandas as pd
import numpy as np
from summer2 import AgeStratification, Overwrite, Multiply
from summer2.parameters import Time, Function
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve import scale_up_function
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.utils.utils import change_parameter_unit
from autumn.models.tb_dynamics.utils import (
    create_sloping_step_function,
    get_parameter_dict_from_function,
    create_step_function_from_dict,
)
from autumn.models.tb_dynamics.constants import Compartment, INFECTIOUS_COMPS
from math import log, exp


def get_age_strat(
    params: Parameters,
    compartments: List[str],
    age_pops: pd.Series = None,
    age_mixing_matrix=None,
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
    iso3 = params.country.iso3
    strat = AgeStratification("age", age_breakpoints, compartments)
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

        step_func = create_step_function_from_dict(latency_params)
        step_func_as_dict = get_parameter_dict_from_function(step_func, params.age_breakpoints)
        adjs = change_parameter_unit(step_func_as_dict, 365.251)
        adjs = {str(k): Overwrite(v) for k, v in adjs.items()}
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

            # if comp == Compartment.ON_TREATMENT:
            #     # Apply infectiousness multiplier for people on treatment,
            #     average_infectiousness *= params.on_treatment_infect_multiplier

            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)
    # Set age-specific treatment recovery, relapse and treatment death rates

    return strat
