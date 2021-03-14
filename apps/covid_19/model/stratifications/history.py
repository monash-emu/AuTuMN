from typing import Dict

from summer import Stratification, Multiply


from apps.covid_19.constants import Compartment, Clinical
from apps.covid_19.model.parameters import Parameters
from apps.covid_19.model.stratifications.clinical import CLINICAL_STRATA
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA
from apps.covid_19.model.preprocess.clinical import (
    get_absolute_strata_proportions,
    get_proportion_symptomatic,
)


def get_history_strat(
    params: Parameters,
    compartment_periods: Dict[str, float],
) -> Stratification:
    """
    Infection history stratification
    """
    history_strat = Stratification(
        "history",
        ["naive", "experienced"],
        [Compartment.SUSCEPTIBLE, Compartment.EARLY_EXPOSED],
    )
    # Everyone starts out naive.
    history_strat.set_population_split({"naive": 1.0, "experienced": 0.0})
    # Waning immunity makes recovered individuals transition to the 'experienced' stratum
    history_strat.add_flow_adjustments(
        "warning_immunity", {"naive": Multiply(0.0), "experienced": Multiply(1.0)}
    )

    # Get the proportion of people in each clinical stratum, relative to total people in compartment.
    clinical_params = params.clinical_stratification
    symptomatic_props = get_proportion_symptomatic(params)
    abs_props = get_absolute_strata_proportions(
        symptomatic_props=symptomatic_props,
        icu_props=clinical_params.icu_prop,
        hospital_props=clinical_params.props.hospital.props,
        symptomatic_props_multiplier=clinical_params.props.symptomatic.multiplier,
        hospital_props_multiplier=clinical_params.props.hospital.multiplier,
    )

    # Adjust parameters defining progression from early exposed to late exposed to obtain the requested proportion
    for age_idx, agegroup in enumerate(AGEGROUP_STRATA):
        # Collect existing rates of progressions for symptomatic vs non-symptomatic
        rate_non_sympt = (
            abs_props[Clinical.NON_SYMPT][age_idx] / compartment_periods[Compartment.EARLY_EXPOSED]
        )
        total_progression_rate = 1.0 / compartment_periods[Compartment.EARLY_EXPOSED]
        rate_sympt = total_progression_rate - rate_non_sympt
        # Multiplier for symptomatic is rel_prop_symptomatic_experienced
        sympt_multiplier = params.rel_prop_symptomatic_experienced
        # Multiplier for asymptomatic rate is 1 + rate_sympt / rate_non_sympt * (1 - sympt_multiplier) in order to preserve aggregated exit flow.
        non_sympt_multiplier = 1 + rate_sympt / rate_non_sympt * (1.0 - sympt_multiplier)
        # Create adjustment requests
        for clinical in CLINICAL_STRATA:
            experienced_multiplier = (
                non_sympt_multiplier if clinical == Clinical.NON_SYMPT else sympt_multiplier
            )
            adjustments = {
                "naive": Multiply(1.0),
                # Sojourn flow, divide by proportion.
                "experienced": Multiply(1.0 / experienced_multiplier),
            }
            history_strat.add_flow_adjustments(
                "infect_onset",
                adjustments,
                dest_strata={"agegroup": agegroup, "clinical": clinical},
            )

    return history_strat
