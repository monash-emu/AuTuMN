import copy

from summer import Overwrite, Stratification

from autumn.models.covid_19.constants import (
    INFECTIOUS_COMPARTMENTS, Clinical, Compartment, CLINICAL_STRATA, PROGRESS, AGE_CLINICAL_TRANSITIONS
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.clinical import get_all_adjustments
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def get_clinical_strat(params: Parameters):
    """
    Stratify the infectious compartments of the covid model by "clinical" status, into five groups.
    """

    clinical_strat = Stratification("clinical", CLINICAL_STRATA, INFECTIOUS_COMPARTMENTS)
    clinical_params = params.clinical_stratification

    """
    Infectiousness adjustments.
    """

    # Start from blank adjustments, then apply to both the late incubation and early active compartments
    non_isolated_adjustments = {stratum: None for stratum in CLINICAL_STRATA}
    non_isolated_adjustments.update({Clinical.NON_SYMPT: Overwrite(clinical_params.non_sympt_infect_multiplier)})
    for compartment in INFECTIOUS_COMPARTMENTS[:-1]:
        clinical_strat.add_infectiousness_adjustments(compartment, non_isolated_adjustments)

    # Pick up where we left for the first two compartments and update for the isolated/hospitalised (last three strata)
    late_active_adjustments = copy.copy(non_isolated_adjustments)
    for stratum in CLINICAL_STRATA[2:]:
        late_active_adj = Overwrite(clinical_params.late_infect_multiplier[stratum])
        late_active_adjustments.update({stratum: late_active_adj})

    # Apply to the compartment
    clinical_strat.add_infectiousness_adjustments(Compartment.LATE_ACTIVE, late_active_adjustments)

    """
    Make all the adjustments to flows.
    """

    # Get all the adjustments in the same way as we will do for the immunity and vaccination stratifications
    adjs = get_all_adjustments(
        clinical_params, params.country, params.population, params.infection_fatality.props, params.sojourn,
        params.infection_fatality.multiplier, params.clinical_stratification.props.symptomatic.multiplier,
        params.clinical_stratification.props.hospital.multiplier, params.infection_fatality.top_bracket_overwrite,
    )

    # Assign all the adjustments to the summer model
    for agegroup in AGEGROUP_STRATA:
        source = {"agegroup": agegroup}
        clinical_strat.add_flow_adjustments(PROGRESS, adjs[PROGRESS], source_strata=source)  # Not age-stratified
        for transition in AGE_CLINICAL_TRANSITIONS:
            clinical_strat.add_flow_adjustments(transition, adjs[transition][agegroup], source_strata=source)

    return clinical_strat
