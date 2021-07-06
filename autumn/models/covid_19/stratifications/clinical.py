from summer import Overwrite, Stratification

from autumn.models.covid_19.constants import INFECTIOUS_COMPARTMENTS, Clinical, Compartment, CLINICAL_STRATA
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.preprocess.clinical import get_all_adjs
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def get_clinical_strat(params: Parameters):

    """
    Stratify the model by clinical status
    Stratify the infectious compartments of the covid model (not including the pre-symptomatic compartments, which are
    actually infectious)

    - notifications are derived from progress from early to late for some strata
    - the proportion of people moving from presympt to early infectious, conditioned on age group
    - rate of which people flow through these compartments (reciprocal of time, using within_* which is a rate of ppl / day)
    - infectiousness levels adjusted by early/late and for clinical strata
    - we start with an age stratified infection fatality rate
        - 50% of deaths for each age bracket die in ICU
        - the other deaths go to hospital, assume no-one else can die from COVID
        - should we ditch this?

    """
    clinical_strat = Stratification("clinical", CLINICAL_STRATA, INFECTIOUS_COMPARTMENTS)
    clinical_params = params.clinical_stratification
    country = params.country
    pop = params.population

    """
    Infectiousness adjustments for clinical stratification
    """
    # Add infectiousness reduction multiplier for all non-symptomatic infectious people.
    # These people are less infectious because of biology.
    non_sympt_adjust = Overwrite(clinical_params.non_sympt_infect_multiplier)
    clinical_strat.add_infectiousness_adjustments(
        Compartment.LATE_EXPOSED,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: None,
            Clinical.ICU: None,
        },
    )
    clinical_strat.add_infectiousness_adjustments(
        Compartment.EARLY_ACTIVE,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.SYMPT_ISOLATE: None,
            Clinical.HOSPITAL_NON_ICU: None,
            Clinical.ICU: None,
        },
    )
    # Add infectiousness reduction for people who are late active and in isolation or hospital/icu.
    # These people are less infectious because of physical distancing/isolation/PPE precautions.
    late_infect_multiplier = clinical_params.late_infect_multiplier
    clinical_strat.add_infectiousness_adjustments(
        Compartment.LATE_ACTIVE,
        {
            Clinical.NON_SYMPT: non_sympt_adjust,
            Clinical.SYMPT_ISOLATE: Overwrite(late_infect_multiplier[Clinical.SYMPT_ISOLATE]),
            Clinical.SYMPT_NON_HOSPITAL: None,
            Clinical.HOSPITAL_NON_ICU: Overwrite(late_infect_multiplier[Clinical.HOSPITAL_NON_ICU]),
            Clinical.ICU: Overwrite(late_infect_multiplier[Clinical.ICU]),
        },
    )

    """
    Adjust infection death rates for hospital patients (ICU and non-ICU)
    """
    symptomatic_adjuster = params.clinical_stratification.props.symptomatic.multiplier
    ifr_adjuster = params.infection_fatality.multiplier
    ifr_top_bracket_overwrite = params.infection_fatality.top_bracket_overwrite

    # This is now unused and could be deleted - was the previous approach for Victoria
    # hospital_adjuster = ifr_adjuster if \
    #     params.clinical_stratification.props.use_ifr_for_severity else \
    #     params.clinical_stratification.props.hospital.multiplier
    hospital_adjuster = params.clinical_stratification.props.hospital.multiplier

    # Get all the adjustments in the same way as we will do if the immunity stratification is implemented
    entry_adjustments, death_adjs, progress_adjs, recovery_adjs, _, get_detected_proportion = get_all_adjs(
        clinical_params,
        country,
        pop,
        params.infection_fatality.props,
        params.sojourn,
        params.testing_to_detection,
        params.case_detection,
        ifr_adjuster,
        symptomatic_adjuster,
        hospital_adjuster,
        ifr_top_bracket_overwrite,
    )

    # Assign all the adjustments to the model
    for agegroup in AGEGROUP_STRATA:
        source = {"agegroup": agegroup}
        clinical_strat.add_flow_adjustments(
            "infect_onset", entry_adjustments[agegroup], source_strata=source
        )
        clinical_strat.add_flow_adjustments(
            "infect_death", death_adjs[agegroup], source_strata=source
        )
        clinical_strat.add_flow_adjustments(
            "progress",
            progress_adjs,
            source_strata=source,
        )
        clinical_strat.add_flow_adjustments(
            "recovery",
            recovery_adjs[agegroup],
            source_strata=source,
        )

    return clinical_strat, get_detected_proportion
