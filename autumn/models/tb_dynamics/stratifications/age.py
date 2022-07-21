from summer import AgeStratification, Overwrite
from autumn.models.tb_dynamics.parameters import Parameters
from autumn.models.tb_dynamics.constants import BASE_COMPARTMENTS
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve import scale_up_function


def get_age_strat(params: Parameters) -> AgeStratification:
    strat = AgeStratification("age", params.age_breakpoints, BASE_COMPARTMENTS)

    death_rates_by_age, death_rate_years = get_death_rates_by_agegroup(
        params.age_breakpoints, params.iso3
    )

    universal_death_funcs = {}
    for age in params.age_breakpoints:
        universal_death_funcs[age] = scale_up_function(
            death_rate_years, death_rates_by_age[age], smoothness=0.2, method=5
        )

    death_adjs = {str(k): Overwrite(v) for k, v in universal_death_funcs.items()}
    print(death_adjs)
    for comp in BASE_COMPARTMENTS:
        flow_name = f"universal_death_for_{comp}"
        strat.set_flow_adjustments(flow_name, death_adjs)

    return strat
