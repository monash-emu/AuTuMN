import copy
from typing import Dict

from summer import Overwrite, Stratification

from autumn.models.covid_19.constants import (
    INFECTIOUS_COMPARTMENTS, Clinical, Compartment, CLINICAL_STRATA, PROGRESS, AGE_CLINICAL_TRANSITIONS
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.strat_processing.clinical import get_all_adjustments
from autumn.settings import COVID_BASE_AGEGROUPS


def get_clinical_strat(params: Parameters, stratified_adjusters: Dict[str, Dict[str, float]]) -> Stratification:
    """
    Stratify the infectious compartments of the covid model by "clinical" status, into the following five groups:

        NON_SYMPT = "non_sympt"
            Asymptomatic persons
        SYMPT_NON_HOSPITAL = "sympt_non_hospital"
            Symptomatic persons who are never detected or admitted to hospital
        SYMPT_ISOLATE = "sympt_isolate"
            Symptomatic persons who are detected by the health system and so may go on to isolate
        HOSPITAL_NON_ICU = "hospital_non_icu"
            Persons with sufficiently severe disease to necessitate admission to hospital, but not to ICU
        ICU = "icu"
            Persons with sufficiently severe disease to necessitate admission to ICU

    Args:
        params: All model parameters
        stratified_adjusters: VoC and severity stratification adjusters

    Returns:
        The clinical stratification summer object for application to the main model

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
    Adjustments to flows.
    """

    for voc in stratified_adjusters.keys():

        # Get all the adjustments in the same way as we will do for the immunity and vaccination stratifications
        adjs = get_all_adjustments(
            clinical_params, params.country, params.population, params.infection_fatality.props, params.sojourn,
            stratified_adjusters[voc]["sympt"], stratified_adjusters[voc]["hosp"], stratified_adjusters[voc]["ifr"],
        )

        # Assign all the adjustments to the summer model
        voc_stratum = {"strain": voc} if params.voc_emergence else {}  # *** Don't filter by VoC if there are no VoCs
        for agegroup in COVID_BASE_AGEGROUPS:
            source = {"agegroup": agegroup}
            source.update(voc_stratum)
            clinical_strat.set_flow_adjustments(PROGRESS, adjs[PROGRESS], source_strata=source)  # Not age-stratified
            for transition in AGE_CLINICAL_TRANSITIONS:
                clinical_strat.set_flow_adjustments(transition, adjs[transition][agegroup], source_strata=source)

    return clinical_strat
