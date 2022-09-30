from typing import List
import pandas as pd
from math import exp, log
from summer import AgeStratification, Overwrite, Multiply
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve import scale_up_function
from autumn.models.tb_dynamics.parameters import Parameters
from autumn.core.utils.utils import change_parameter_unit
from autumn.models.tuberculosis.constants import Compartment, INFECTIOUS_COMPS
from autumn.models.tb_dynamics.utils import (
    create_sloping_step_function,
    get_parameter_dict_from_function,
    create_step_function_from_dict,
)

def get_age_strat( 
    params: Parameters,
    age_pops: pd.Series,
    compartments: List[str],
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

     # Set age-specific infectiousness
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
    time_variant_tsr = scale_up_function(
        list(params.time_variant_tsr.keys()), list(params.time_variant_tsr.values()), method=4
    )

    # Set treatment_recovery
    treatment_recovery_funcs = {}
    treatment_death_funcs = {}
    treatment_relapse_funcs = {}
    for age in params.age_breakpoints:
        def get_treatment_recovery_rate(t, computed_values, age=age):
            death_rate = universal_death_funcs[age](t)
            floor_val = 1 / params.treatment_duration
            dynamic_val = (
                death_rate
                / params.prop_death_among_negative_tx_outcome
                * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0)
            )
            return max(floor_val, dynamic_val)

        def get_treatment_death_rate(t, computed_values, age=age):
            death_rate = universal_death_funcs[age](t)
            recovery_rate = get_treatment_recovery_rate(t, computed_values, age=age)
            return (
                params.prop_death_among_negative_tx_outcome
                * recovery_rate
                * (1.0 - time_variant_tsr(t))
                / time_variant_tsr(t)
                - death_rate
            )

        def get_treatment_relapse_rate(t, computed_values, age=age):
            recovery_rate = get_treatment_recovery_rate(t, computed_values, age=age)
            return (
                recovery_rate
                * (1.0 / time_variant_tsr(t) - 1.0)
                * (1.0 - params.prop_death_among_negative_tx_outcome)
            )

        treatment_recovery_funcs[age] = get_treatment_recovery_rate
        treatment_death_funcs[age] = get_treatment_death_rate
        treatment_relapse_funcs[age] = get_treatment_relapse_rate

    treatment_recovery_adjs = {str(k): Multiply(v) for k, v in treatment_recovery_funcs.items()}
    treatment_death_adjs = {str(k): Multiply(v) for k, v in treatment_death_funcs.items()}
    treatment_relapse_adjs = {str(k): Multiply(v) for k, v in treatment_relapse_funcs.items()}
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_adjs)
    strat.set_flow_adjustments("treatment_death", treatment_death_adjs)
    strat.set_flow_adjustments("relapse", treatment_relapse_adjs)

    # Add BCG effect without stratifying for BCG
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    bcg_multilier_dict = get_parameter_dict_from_function(bcg_wane, params.age_breakpoints)
    bcg_coverage_func = scale_up_function(
        list(params.time_variant_bcg_perc.keys()),
        list(params.time_variant_bcg_perc.values()),
        method=5,
        bound_low=0,
        bound_up=100,
        smoothness=1.5,
    )
    bcg_adjs = {}
    for age, multiplier in bcg_multilier_dict.items():
        if multiplier < 1.0:
            average_age = get_average_age_for_bcg(age, params.age_breakpoints)
            bcg_adjs[str(age)] = Multiply(
                make_bcg_multiplier_func(bcg_coverage_func, multiplier, average_age)
            )
        else:
            bcg_adjs[str(age)] = None

    if params.bcg_effect == "infection":
        flow_affected_by_bcg = "infection"
    elif params.bcg_effect == "mortality":
        flow_affected_by_bcg = "infect_death"
    strat.set_flow_adjustments(flow_affected_by_bcg, bcg_adjs)

    return strat

def get_average_age_for_bcg(agegroup, age_breakpoints):
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print("Warning: the agegroup name is being used to represent the average age of the group")
        return float(agegroup)
    else:
        return 0.5 * (age_breakpoints[agegroup_idx] + age_breakpoints[agegroup_idx + 1])


def make_bcg_multiplier_func(bcg_coverage_func, multiplier, average_age):
    def bcg_multiplier_func(t, computed_values):
        return 1.0 - bcg_coverage_func(t - average_age) / 100.0 * (1.0 - multiplier)

    return bcg_multiplier_func
