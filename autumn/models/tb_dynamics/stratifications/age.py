from typing import List
import pandas as pd
import numpy as np
from summer import AgeStratification, Overwrite
from autumn.core.inputs import get_death_rates_by_agegroup
from autumn.model_features.curve import scale_up_function


def get_age_strat( 
    age_breakpoints: List[str],
    iso3: str,
    age_pops: pd.Series,
    age_mixing_matrix,
    compartments: List[str]
) -> AgeStratification:

    """Stratify the model by age

    Returns:
        Stratification object
    """
    strat = AgeStratification("age", age_breakpoints, compartments)
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

    return strat
