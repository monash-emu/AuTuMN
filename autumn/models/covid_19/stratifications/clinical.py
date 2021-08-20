import copy

from summer import Overwrite, Stratification

from autumn.models.covid_19.constants import (
    INFECTIOUS_COMPARTMENTS,
    Clinical,
    Compartment,
    CLINICAL_STRATA,
    INFECTIOUSNESS_ONSET,
    PROGRESS,
    INFECT_DEATH,
    RECOVERY,
)
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.clinical import get_all_adjustments
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def get_clinical_strat(params: Parameters):
    """
    Stratify the infectious compartments of the covid model by "clinical" status.
    """

    clinical_strat = Stratification("clinical", CLINICAL_STRATA, INFECTIOUS_COMPARTMENTS)
    clinical_params = params.clinical_stratification

    """
    Infectiousness adjustments
    """

    # Start from blank adjustments because all strata must be specified by summer rules
    non_isolated_adjustments = {stratum: None for stratum in CLINICAL_STRATA}
    non_isolated_adjustments.update({Clinical.NON_SYMPT: Overwrite(clinical_params.non_sympt_infect_multiplier)})

    # Apply this to both the late incubation and early active periods
    for compartment in (Compartment.LATE_EXPOSED, Compartment.EARLY_ACTIVE):
        clinical_strat.add_infectiousness_adjustments(compartment, non_isolated_adjustments)

    # Start from where we left off for the late incubation and early active periods - including the asymptomatic
    late_active_adjustments = copy.copy(non_isolated_adjustments)

    # Update the ones in late active who are isolated or admitted to hospital
    for stratum in [Clinical.SYMPT_ISOLATE, Clinical.HOSPITAL_NON_ICU, Clinical.ICU]:
        late_active_adjustments.update(
            {stratum: Overwrite(clinical_params.late_infect_multiplier[stratum])}
        )

    # Apply to the compartment
    clinical_strat.add_infectiousness_adjustments(Compartment.LATE_ACTIVE, late_active_adjustments)

    """
    Adjust infection death rates for hospital patients (ICU and non-ICU)
    """

    # Get all the adjustments in the same way as we will do for the immunity and vaccination stratifications
    entry_adjs, death_adjs, progress_adjs, recovery_adjs, get_detected_proportion, adj_systems = get_all_adjustments(
        clinical_params,
        params.country,
        params.population,
        params.infection_fatality.props,
        params.sojourn,
        params.testing_to_detection,
        params.case_detection, params.infection_fatality.multiplier,
        params.clinical_stratification.props.symptomatic.multiplier,
        params.clinical_stratification.props.hospital.multiplier,
        params.infection_fatality.top_bracket_overwrite,
    )

    # Assign all the adjustments to the summer model
    for agegroup in AGEGROUP_STRATA:
        source = {"agegroup": agegroup}
        clinical_strat.add_flow_adjustments(INFECTIOUSNESS_ONSET, entry_adjs[agegroup], source_strata=source)
        clinical_strat.add_flow_adjustments(PROGRESS, progress_adjs, source_strata=source)
        clinical_strat.add_flow_adjustments(INFECT_DEATH, death_adjs[agegroup], source_strata=source)
        clinical_strat.add_flow_adjustments(RECOVERY, recovery_adjs[agegroup], source_strata=source)

    return clinical_strat, get_detected_proportion, adj_systems
