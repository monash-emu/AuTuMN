from summer2 import AgeStratification, Overwrite, Multiply
from summer2.parameters import Time, Function
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve.interpolate import build_static_sigmoidal_multicurve
from autumn.core.inputs.social_mixing.build_synthetic_matrices import build_synthetic_matrices
from autumn.models.tb_dynamics2.constants import Compartment, INFECTIOUS_COMPS
from autumn.models.tb_dynamics2.constants import BASE_COMPARTMENTS
from autumn.models.tb_dynamics2.utils import *

def get_age_strat(
    params,
) -> AgeStratification:

    """
    Function to create the age group stratification object..

    Args:
        params: Parameter class
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
        #is_activation_flow = flow_name in ["late_activation"]
        #if is_activation_flow:
        adjs = {str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])]) for t in age_breaks}
        strat.set_flow_adjustments(flow_name, adjs)

    # is_activation_flow = flow_name in ["early_activation", "late_activation"]

<<<<<<< HEAD
    # if params.inflate_reactivation_for_diabetes and is_activation_flow:
    #         # Inflate reactivation rate to account for diabetes.
    #         for age in params.age_breakpoints:
    #             adjs[age] = Function(get_latency_with_diabetes, [Time, params.prop_diabetes[age], adjs[str(age)],params.rr_progression_diabetes])
=======
    if params.inflate_reactivation_for_diabetes and is_activation_flow:
            # Inflate reactivation rate to account for diabetes.
            for age in params.age_breakpoints:
                adjs[age] = Function(get_latency_with_diabetes, [Time, 
                                                                params.prop_diabetes[age], 
                                                                adjs[str(age)], 
                                                                params.rr_progression_diabetes])
>>>>>>> 031fa319bdc9f114c5f7da498eab596132ef9076
        
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
        [Time]
    )

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
    bcg_multiplier_dict = {'0': 0.3, '5': 0.3, '15': 0.7375, '35': 1.0, '50': 1.0} # Ragonnet et al. (IJE, 2020)
    bcg_coverage_func = build_static_sigmoidal_multicurve(
        list(params.time_variant_bcg_perc.keys()),
        list(params.time_variant_bcg_perc.values()),
    )
    bcg_adjs = {}
    for age, multiplier in bcg_multiplier_dict.items():
        if multiplier < 1.0:
            bcg_adjs[str(age)] = Multiply(
                Function(bcg_multiplier_func, [Time, bcg_coverage_func, multiplier, get_average_age_for_bcg(age, age_breaks)])
            )
        else:
            bcg_adjs[str(age)] = None
    if params.bcg_effect == "infection":
        flow_affected_by_bcg = "infection"
    elif params.bcg_effect == "mortality":
        flow_affected_by_bcg = "infect_death"
    strat.set_flow_adjustments(flow_affected_by_bcg, bcg_adjs)

    return strat
