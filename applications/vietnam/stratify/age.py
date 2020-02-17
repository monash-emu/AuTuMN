from summer_py.summer_model.utils.parameter_processing import (
    get_parameter_dict_from_function,
    logistic_scaling_function,
)
from summer_py.summer_model import (
    StratifiedModel,
    split_age_parameter,
    create_sloping_step_function,
)
from autumn.tb_model import get_adapted_age_parameters
from autumn.db import Database, get_pop_mortality_functions


def stratify_age(model: StratifiedModel, input_database: Database, age_params: dict):
    age_breakpoints = age_params["strata"]
    adult_latency_adjustment = age_params["adult_latency_adjustment"]
    ipt_ct_coverage = age_params["ipt_ct_coverage"]

    age_infectiousness = get_parameter_dict_from_function(
        logistic_scaling_function(10.0), age_breakpoints
    )
    age_params = get_adapted_age_parameters(age_breakpoints)
    age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

    # Adjustment of latency parameters
    for param in ["early_progression", "late_progression"]:
        for age_break in age_breakpoints:
            if age_break > 5:
                age_params[param][str(age_break) + "W"] *= adult_latency_adjustment

    pop_morts = get_pop_mortality_functions(input_database, age_breakpoints, country_iso_code="VNM")

    age_params["universal_death_rate"] = {}
    for age_break in age_breakpoints:
        model.time_variants[f"universal_death_rateXage_{age_break}"] = pop_morts[age_break]
        model.parameters[
            f"universal_death_rateXage_{age_break}"
        ] = f"universal_death_rateXage_{age_break}"

        age_params["universal_death_rate"][
            f"{age_break}W"
        ] = f"universal_death_rateXage_{age_break}"

    model.parameters["universal_death_rateX"] = 0.0

    # Age-specific IPT
    age_params.update({"ipt_rate": ipt_ct_coverage})

    # Add BCG effect without stratification assuming constant 100% coverage
    bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
    age_bcg_efficacy_dict = get_parameter_dict_from_function(
        lambda value: bcg_wane(value), age_breakpoints
    )
    age_params.update({"contact_rate": age_bcg_efficacy_dict})

    model.stratify(
        "age",
        age_breakpoints,
        [],
        {},
        adjustment_requests=age_params,
        infectiousness_adjustments=age_infectiousness,
        verbose=False,
    )

    # FIXME: Does this still need to be used?
    # patch for IPT to overwrite parameters when ds_ipt has been turned off while we still need some coverage at baseline
    # if external_params["ds_ipt_switch"] == 0.0 and external_params["mdr_ipt_switch"] == 1.0:
    #     model.parameters["ipt_rateXstrain_dsXage_0"] = 0.17
    #     for age_break in [5, 15, 60]:
    #         model.parameters["ipt_rateXstrain_dsXage_" + str(age_break)] = 0.0
