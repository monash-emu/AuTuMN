from summer2 import AgeStratification, Overwrite, Multiply
from summer2.parameters import Time, Function
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices

from autumn.models.tb_dynamics2.constants import Compartment, INFECTIOUS_COMPS

from math import log, exp

from jax import numpy as jnp
from autumn.models.tb_dynamics2.constants import BASE_COMPARTMENTS


def get_treatment_outcomes(duration, prop_death_among_non_success, natural_death_rate, tsr):
    
    # Calculate the proportion of people dying from natural causes while on treatment
    prop_natural_death_while_on_treatment = 1.0 - jnp.exp(-duration * natural_death_rate)

    # Calculate the target proportion of treatment outcomes resulting in death based on requests
    requested_prop_death_on_treatment = (1.0 - tsr) * prop_death_among_non_success
    
    # Calculate the actual rate of deaths on treatment, with floor of zero
    prop_death_from_treatment = jnp.max(jnp.array((requested_prop_death_on_treatment - prop_natural_death_while_on_treatment, 0.0)))
    
    # Calculate the proportion of treatment episodes resulting in relapse
    relapse_prop = 1.0 - tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    
    return tuple([param * duration for param in [tsr, prop_death_from_treatment, relapse_prop]])


def get_average_sigmoid(low_val, upper_val, inflection):
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
    """
    return (log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))) / (upper_val - low_val)


def map_params_to_model_agegroups(input_dict, targets):
    results = {}
    for t in targets:
        results[str(t)] = Multiply(input_dict[max([k for k in input_dict.keys() if k <= t])])
    return results

        
def get_average_age_for_bcg(agegroup, age_breakpoints):
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print("Warning: the agegroup name is being used to represent the average age of the group")
        return float(agegroup)
    else:
        return 0.5 * (age_breakpoints[agegroup_idx] + age_breakpoints[agegroup_idx + 1])


def bcg_multiplier_func(t, tfunc, fmultiplier, faverage_age):
    return 1.0 - tfunc(t - faverage_age) / 100.0 * (1.0 - fmultiplier)


def get_age_strat(
    params,
) -> AgeStratification:

    """
    Function to create the age group stratification object..

    Args:
        params: Parameter class
        age_pops: The population distribution by age
        compartments: All the model compartments
    Returns:
        The age stratification summer object
    """
    age_breaks = params.age_breakpoints
    iso3 = params.country.iso3
    strat = AgeStratification("age", age_breaks, BASE_COMPARTMENTS)

    # Get and set age-specific mixing matrix
    age_mixing_matrix = build_synthetic_matrices(
        iso3,
        params.age_mixing.source_iso3,
        params.age_breakpoints,
        params.age_mixing.age_adjust,
        requested_locations=["all_locations"],
    )["all_locations"]
    age_mixing_matrix *= 365.251
    strat.set_mixing_matrix(age_mixing_matrix)

    # Set non-TB-related mortality rates
    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(age_breaks, iso3)
    universal_death_funcs, death_adjs = {}, {}
    for age in age_breaks:
        age_specific_death_func = build_static_sigmoidal_multicurve(death_rate_years, death_rates_by_age[age])
        universal_death_funcs[age] = Function(age_specific_death_func, [Time])
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)

    # Set age-specific late activation rate
    for flow_name, latency_params in params.age_stratification.items():
        strat.set_flow_adjustments(flow_name, map_params_to_model_agegroups(latency_params, age_breaks))

    # Increasing infectiousness with age
    inf_switch_age = params.age_infectiousness_switch
    for comp in INFECTIOUS_COMPS:
        inf_adjs = {}
        for i, age_low in enumerate(age_breaks):
            infectiousness = 1.0 if age_low == age_breaks[-1] else get_average_sigmoid(age_low, age_breaks[i + 1], inf_switch_age)

            # Infectiousness multiplier for treatment (ideally move to model.py, but has to be set in stratification with current summer)
            if comp == Compartment.ON_TREATMENT:
                infectiousness *= params.on_treatment_infect_multiplier

            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Get the time-varying treatment success proportions
    time_variant_tsr = Function(
        build_static_sigmoidal_multicurve(
            list(params.time_variant_tsr.keys()), list(params.time_variant_tsr.values())
        ),
        [Time],
    ) # Scaling up the time-variant treatment success rate based on observed values

    # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = {}, {}, {}
    for age in age_breaks:
        death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            get_treatment_outcomes,
            [
                params.treatment_duration,
                params.prop_death_among_negative_tx_outcome,
                death_rate,
                time_variant_tsr,
            ],
        )
        treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
        treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
        treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    strat.set_flow_adjustments("relapse", treatment_relapse_funcs)

    # Add BCG effect without stratifying for BCG
    bcg_multiplier_dict = {'0': 0.3, '5': 0.3, '15': 0.7375, '35': 1.0, '50': 1.0}
    bcg_coverage_func = build_static_sigmoidal_multicurve(
        list(params.time_variant_bcg_perc.keys()),
        list(params.time_variant_bcg_perc.values()),
    )
    bcg_adjs = {}
    for age, multiplier in bcg_multiplier_dict.items():
        if multiplier < 1.0:
            average_age = get_average_age_for_bcg(age, age_breaks)
            bcg_adjs[age] = Multiply(
                Function(bcg_multiplier_func, [Time, bcg_coverage_func, multiplier, average_age])
            )
        else:
            bcg_adjs[str(age)] = None
    if params.bcg_effect == "infection":
        flow_affected_by_bcg = "infection"
    elif params.bcg_effect == "mortality":
        flow_affected_by_bcg = "infect_death"
    strat.set_flow_adjustments(flow_affected_by_bcg, bcg_adjs)

    return strat
