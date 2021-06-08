from summer import Stratification, Multiply
from math import exp, log

from autumn.models.tuberculosis.constants import Compartment, COMPARTMENTS, INFECTIOUS_COMPS
from autumn.models.tuberculosis.parameters import Parameters
from autumn.tools.inputs.social_mixing.queries import get_mixing_matrix_specific_agegroups


from summer.legacy.model import create_sloping_step_function
from summer.legacy.model.utils.parameter_processing import (
    get_parameter_dict_from_function,
)

from autumn.tools.curve import make_linear_curve, scale_up_function, tanh_based_scaleup
from autumn.tools.inputs import get_death_rates_by_agegroup
from autumn.tools.inputs.social_mixing.queries import get_mixing_matrix_specific_agegroups

from summer.legacy.model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
)

from autumn.tools.curve import make_linear_curve, tanh_based_scaleup
from autumn.tools.utils.utils import change_parameter_unit


def get_age_strat(params: Parameters) -> Stratification:
    strat = Stratification("agegroup", params.age_breakpoints, COMPARTMENTS)

    mixing_matrix = get_mixing_matrix_specific_agegroups(params.iso3, params.age_breakpoints)
    strat.set_mixing_matrix(mixing_matrix)

    # Add an age-specific all-causes mortality rate
    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(
        params.age_breakpoints, params.iso3
    )
    death_adjs = {}
    for age in params.age_breakpoints:
        death_adjs[age] = scale_up_function(
            death_rate_years, death_rates_by_age[age], smoothness=0.2, method=5
        )

    strat.add_flow_adjustments("universal_death", death_adjs)

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

        if params.inflate_reactivation_for_diabetes and is_activation_flow:
            # Inflate reactivation rate to account for diabetes.
            diabetes_scale_up = tanh_based_scaleup(
                shape=0.05, inflection_time=1980, lower_asymptote=0.0, upper_asymptote=1.0
            )
            future_diabetes_trend = make_linear_curve(
                x_0=2020, x_1=2050, y_0=1, y_1=params.extra_params["future_diabetes_multiplier"]
            )

            def combined_diabetes_scale_up(t):
                multiplier = future_diabetes_trend(t) if t > 2020 else 1.0
                return multiplier * diabetes_scale_up(t)

            for age in params.age_breakpoints:

                def get_latency_with_diabetes(
                    t,
                    prop_diabetes=params.extra_params["prop_diabetes"][age],
                    previous_progression_rate=adjs[age],
                    rr_progression_diabetes=params.extra_params["rr_progression_diabetes"],
                ):
                    return (
                        1.0
                        - combined_diabetes_scale_up(t)
                        * prop_diabetes
                        * (1.0 - rr_progression_diabetes)
                    ) * previous_progression_rate

                adjs[age] = get_latency_with_diabetes

        adjs = {k: Multiply(v) for k, v in adjs.items()}
        strat.add_flow_adjustments(flow_name, adjs)

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

            inf_adjs[age_low] = Multiply(average_infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Set age-specific treatment recovery, relapse and treatment death rates
    time_variant_tsr = scale_up_function(
        list(params.time_variant_tsr.keys()), list(params.time_variant_tsr.values()), method=4
    )

    # Set treatment_recovery
    def make_treatment_recovery_func(age_group, model, params, time_variant_tsr):
        def treatment_recovery_func(t):
            return max(
                1 / params["treatment_duration"],
                model.time_variants["universal_death_rate_" + str(age_group)](t)
                / params["prop_death_among_negative_tx_outcome"]
                * (1.0 / (1.0 - time_variant_tsr(t)) - 1.0),
            )

        return treatment_recovery_func

    # Set treatment_death
    # Set relapse

    factory_functions = {
        "treatment_recovery_rate": make_treatment_recovery_func,
        "treatment_death_rate": make_treatment_death_func,
        "relapse_rate": make_relapse_rate_func,
    }
    for param_stem in factory_functions:
        flow_adjustments[param_stem] = {}
        for age_group in params["age_breakpoints"]:
            flow_adjustments[param_stem][str(age_group)] = param_stem + "_" + str(age_group)

            model.time_variants[param_stem + "_" + str(age_group)] = factory_functions[param_stem](
                age_group, model, params, time_variant_tsr
            )
            model.parameters[param_stem + "_" + str(age_group)] = param_stem + "_" + str(age_group)

    # ================================

    # add BCG effect without stratifying for BCG
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    bcg_susceptibility_multilier_dict = get_parameter_dict_from_function(
        lambda value: bcg_wane(value), params["age_breakpoints"]
    )
    bcg_coverage_func = scale_up_function(
        list(params["time_variant_bcg_perc"].keys()),
        list(params["time_variant_bcg_perc"].values()),
        method=5,
        bound_low=0,
        bound_up=100,
        smoothness=1.5,
    )
    for agegroup, multiplier in bcg_susceptibility_multilier_dict.items():
        if multiplier < 1.0:
            average_age = get_average_age_for_bcg(agegroup, params["age_breakpoints"])
            name = "contact_rate_" + agegroup
            bcg_susceptibility_multilier_dict[agegroup] = name
            model.time_variants[name] = make_bcg_multiplier_func(
                bcg_coverage_func, multiplier, average_age
            )
            model.parameters[name] = name
    flow_adjustments.update({"contact_rate": bcg_susceptibility_multilier_dict})

    # ================================

    return strat
