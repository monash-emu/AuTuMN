from summer import CompartmentalModel

from autumn.models.covid_19.constants import NOTIFICATION_CLINICAL_STRATA, Clinical, NOTIFICATIONS, INCIDENCE, PROGRESS
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.models.covid_19.stratifications.tracing import Tracing


def request_stratified_output_for_flow(
        model, flow, strata, stratification, name_stem=None, filter_on="destination"
):
    """
    Standardise looping over stratum to pull out stratified outputs for flow.
    """

    stem = name_stem if name_stem else flow
    for stratum in strata:
        if filter_on == "destination":
            model.request_output_for_flow(
                name=f"{stem}X{stratification}_{stratum}",
                flow_name=flow,
                dest_strata={stratification: stratum},
            )
        elif filter_on == "source":
            model.request_output_for_flow(
                name=f"{stem}X{stratification}_{stratum}",
                flow_name=flow,
                source_strata={stratification: stratum},
            )
        else:
            raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


def request_double_stratified_output_for_flow(
        model, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None, filter_on="destination"
):
    """
    As for previous function, but looping over two stratifications.
    """

    stem = name_stem if name_stem else flow
    for stratum_1 in strata_1:
        for stratum_2 in strata_2:
            name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
            if filter_on == "destination":
                model.request_output_for_flow(
                    name=name,
                    flow_name=flow,
                    dest_strata={
                        stratification_1: stratum_1,
                        stratification_2: stratum_2,
                    }
                )
            elif filter_on == "source":
                model.request_output_for_flow(
                    name=name,
                    flow_name=flow,
                    source_strata={
                        stratification_1: stratum_1,
                        stratification_2: stratum_2,
                    }
                )
            else:
                raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


def request_stratified_output_for_compartment(
        model, request_name, compartments, strata, stratification, save_results=True
):
    for stratum in strata:
        full_request_name = f"{request_name}X{stratification}_{stratum}"
        model.request_output_for_compartments(
            name=full_request_name,
            compartments=compartments,
            strata={stratification: stratum},
            save_results=save_results,
        )


def request_standard_outputs(model: CompartmentalModel, params: Parameters):
    """
    Request all of the standard model outputs for the COVID-19 model, other than some specific ones used in Victoria.
    """

    # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
    if params.contact_tracing:
        non_notified_clinical_strata = [strat for strat in CLINICAL_STRATA if strat not in NOTIFICATION_CLINICAL_STRATA]
        for agegroup in AGEGROUP_STRATA:
            for clinical in non_notified_clinical_strata:
                name = f"{INCIDENCE}_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name=INCIDENCE,
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": Tracing.TRACED},
                    save_results=False,
                )

    """
    Healthcare admissions
    """

    notification_sources = [
        f"{PROGRESS}Xagegroup_{age}Xclinical_{clinical}"
        for age in AGEGROUP_STRATA
        for clinical in NOTIFICATION_CLINICAL_STRATA
    ]

    # We also need to capture traced cases that are not already captured with NOTIFICATION_CLINICAL_STRATA
    notifications_traced_by_age_sources = {}
    if params.contact_tracing:
        for agegroup in AGEGROUP_STRATA:
            notifications_traced_by_age_sources[agegroup] = []
            for clinical in [s for s in CLINICAL_STRATA if s not in NOTIFICATION_CLINICAL_STRATA]:
                name = f"{PROGRESS}_tracedXagegroup_{agegroup}Xclinical_{clinical}"
                model.request_output_for_flow(
                    name=name,
                    flow_name=PROGRESS,
                    dest_strata={"agegroup": agegroup, "clinical": clinical, "tracing": Tracing.TRACED},
                    save_results=False,
                )
                notification_sources.append(name)
                notifications_traced_by_age_sources[agegroup].append(name)

    model.request_aggregate_output(name=NOTIFICATIONS, sources=notification_sources)

    # Cumulative incidence and notifications
    if params.cumul_incidence_start_time:
        for existing_output in [INCIDENCE, NOTIFICATIONS]:
            model.request_cumulative_output(
                name=f"accum_{existing_output}",
                source=existing_output,
                start_time=params.cumul_incidence_start_time,
            )

    # Notifications by age group
    for agegroup in AGEGROUP_STRATA:
        sympt_isolate_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.SYMPT_ISOLATE}"
        hospital_non_icu_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}"
        icu_name = f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}"
        notifications_by_age_sources = [sympt_isolate_name, hospital_non_icu_name, icu_name]
        if params.contact_tracing:
            notifications_by_age_sources += notifications_traced_by_age_sources[agegroup]

        model.request_aggregate_output(
            name=f"{NOTIFICATIONS}Xagegroup_{agegroup}",
            sources=notifications_by_age_sources
        )

