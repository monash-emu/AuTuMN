from summer import CompartmentalModel

from autumn.models.covid_19.constants import (
    INFECT_DEATH, INFECTION, Compartment, PROGRESS, NOTIFICATIONS, NOTIFICATION_CLINICAL_STRATA, INCIDENCE,
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.settings import Region
from autumn.models.covid_19.parameters import Parameters


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


def request_common_outputs(model: CompartmentalModel, params: Parameters, is_region_vic):

    # Clusters to cycle over for Vic model if needed
    clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS] if is_region_vic else None

    """
    Incidence
    """

    # Unstratified
    model.request_output_for_flow(name=INCIDENCE, flow_name=INCIDENCE)

    # Stratified by age group
    request_stratified_output_for_flow(model, INCIDENCE, AGEGROUP_STRATA, "agegroup")

    # Stratified by age group and by clinical stratum
    request_double_stratified_output_for_flow(
        model, INCIDENCE, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical"
    )

    # Cumulative incidence
    if params.cumul_incidence_start_time:
        model.request_cumulative_output(
            name=f"accum_{INCIDENCE}",
            source=INCIDENCE,
            start_time=params.cumul_incidence_start_time,
        )

    """
    Infection
    """

    # susceptible_infection_rate functions only work for SEIR structure, would need to change for SEIRS, SEIS, etc.
    model.request_output_for_flow("infection", INFECTION)
    model.request_output_for_compartments("_susceptible", [Compartment.SUSCEPTIBLE], save_results=False)
    model.request_function_output(
        "susceptible_infection_rate",
        func=lambda infection, susceptible: infection / susceptible,
        sources=[INFECTION, "_susceptible"]
    )

    """
    Notifications
    """

    # Unstratified
    notification_pathways = []

    # First track all traced cases (regardless of clinical stratum)
    if params.contact_tracing:
        name = "progress_traced"
        notification_pathways.append(name)
        model.request_output_for_flow(
            name=name,
            flow_name="progress",
            dest_strata={"tracing": "traced"},
            save_results=False,
        )

    # Then track untraced cases that are passively detected (depending on clinical stratum)
    for clinical in NOTIFICATION_CLINICAL_STRATA:
        name = f"progress_untracedX{clinical}"
        dest_strata = {"clinical": clinical, "tracing": "untraced"} if \
            params.contact_tracing else \
            {"clinical": clinical}
        notification_pathways.append(name)
        model.request_output_for_flow(
            name=name,
            flow_name="progress",
            dest_strata=dest_strata,
            save_results=False,
        )
    model.request_aggregate_output(name="notifications", sources=notification_pathways)

    age_notification_pathways = {}
    for agegroup in AGEGROUP_STRATA:
        age_notification_pathways[agegroup] = []

        # First track all traced cases (regardless of clinical stratum)
        if params.contact_tracing:
            name = f"progress_tracedX{agegroup}"
            age_notification_pathways[agegroup].append(name)
            model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata={"tracing": "traced", "agegroup": agegroup},
                save_results=False,
            )

        # Then track untraced cases that are passively detected (depending on clinical stratum)
        for clinical in NOTIFICATION_CLINICAL_STRATA:
            name = f"progress_untracedXagegroup_{agegroup}X{clinical}"
            dest_strata = {"clinical": clinical, "tracing": "untraced", "agegroup": agegroup} if \
                params.contact_tracing else \
                {"clinical": clinical, "agegroup": agegroup}
            age_notification_pathways[agegroup].append(name)
            model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata=dest_strata,
                save_results=False,
            )
        model.request_aggregate_output(
            name=f"notificationsXagegroup_{agegroup}", sources=age_notification_pathways[agegroup]
        )

    # Cumulative notifications
    if params.cumul_incidence_start_time:
        model.request_cumulative_output(
            name=f"accum_{NOTIFICATIONS}",
            source=NOTIFICATIONS,
            start_time=params.cumul_incidence_start_time,
        )

    """
    Case detection
    """

    model.request_computed_value_output("cdr")

    """
    Progression
    """

    # Unstratified
    model.request_output_for_flow(name=PROGRESS, flow_name=PROGRESS)

    # Stratified by age group and clinical status
    request_double_stratified_output_for_flow(
        model, PROGRESS, AGEGROUP_STRATA, "agegroup", NOTIFICATION_CLINICAL_STRATA, "clinical"
    )

    # Stratified by cluster
    if is_region_vic:
        request_stratified_output_for_flow(
            model, PROGRESS,
            [region.replace("-", "_") for region in Region.VICTORIA_SUBREGIONS],
            "cluster", "progress_for_", filter_on="destination",
        )

    """
    Deaths
    """

    # Unstratified
    model.request_output_for_flow(name="infection_deaths", flow_name=INFECT_DEATH)
    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

    # Stratified by age
    request_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem="infection_deaths", filter_on="source"
    )
    for agegroup in AGEGROUP_STRATA:
        model.request_cumulative_output(
            name=f"accum_deathsXagegroup_{agegroup}",
            source=f"infection_deathsXagegroup_{agegroup}",
        )

    # Stratified by age and clinical stratum
    request_double_stratified_output_for_flow(
        model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup",
        CLINICAL_STRATA, "clinical", name_stem="infection_deaths", filter_on="source"
    )

    # Victoria-specific stratification by cluster
    if is_region_vic:
        for cluster in clusters:
            model.request_output_for_flow(
                name=f"infection_deaths_for_cluster_{cluster}",
                flow_name=INFECT_DEATH,
                source_strata={"cluster": cluster},
            )
