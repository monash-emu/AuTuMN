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


class Outputs:
    def __init__(self, model):
        self.model = model

    def request_stratified_output_for_flow(self, flow, strata, stratification, name_stem=None, filter_on="destination"):
        """
        Standardise looping over stratum to pull out stratified outputs for flow.
        """

        stem = name_stem if name_stem else flow
        for stratum in strata:
            if filter_on == "destination":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    dest_strata={stratification: stratum},
                )
            elif filter_on == "source":
                self.model.request_output_for_flow(
                    name=f"{stem}X{stratification}_{stratum}",
                    flow_name=flow,
                    source_strata={stratification: stratum},
                )
            else:
                raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")

    def request_double_stratified_output_for_flow(
            self, flow, strata_1, stratification_1, strata_2, stratification_2, name_stem=None, filter_on="destination"
    ):
        """
        As for previous function, but looping over two stratifications.
        """

        stem = name_stem if name_stem else flow
        for stratum_1 in strata_1:
            for stratum_2 in strata_2:
                name = f"{stem}X{stratification_1}_{stratum_1}X{stratification_2}_{stratum_2}"
                if filter_on == "destination":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        dest_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                elif filter_on == "source":
                    self.model.request_output_for_flow(
                        name=name,
                        flow_name=flow,
                        source_strata={
                            stratification_1: stratum_1,
                            stratification_2: stratum_2,
                        }
                    )
                else:
                    raise ValueError(f"filter_on should be either 'source' or 'destination', found {filter_on}")


class CovidOutputs(Outputs):

    def request_incidence(self):
        """
        Request incidence rate calculations.
        """

        # Unstratified
        self.model.request_output_for_flow(name=INCIDENCE, flow_name=INCIDENCE)

        # Stratified by age group
        self.request_stratified_output_for_flow(INCIDENCE, AGEGROUP_STRATA, "agegroup")

        # Stratified by age group and by clinical stratum
        self.request_double_stratified_output_for_flow(
            INCIDENCE, AGEGROUP_STRATA, "agegroup", CLINICAL_STRATA, "clinical"
        )

    def request_infection(self):
        """
        Request infection rate calculations.

        Note that susceptible_infection_rate functions only work for SEIR structure, would need to change for SEIRS,
        SEIS, etc.
        """

        self.model.request_output_for_flow("infection", INFECTION)

    def request_notifications(self, contact_tracing_params, cumul_inc_start_time):

        # Unstratified
        notification_pathways = []

        # First track all traced cases (regardless of clinical stratum)
        if contact_tracing_params:
            name = "progress_traced"
            notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata={"tracing": "traced"},
                save_results=False,
            )

        # Then track untraced cases that are passively detected (depending on clinical stratum)
        for clinical in NOTIFICATION_CLINICAL_STRATA:
            name = f"progress_untracedX{clinical}"
            dest_strata = {"clinical": clinical, "tracing": "untraced"} if \
                contact_tracing_params else \
                {"clinical": clinical}
            notification_pathways.append(name)
            self.model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata=dest_strata,
                save_results=False,
            )
        self.model.request_aggregate_output(name="notifications", sources=notification_pathways)

        # Cumulative unstratified notifications
        if cumul_inc_start_time:
            self.model.request_cumulative_output(
                name=f"accum_{NOTIFICATIONS}",
                source=NOTIFICATIONS,
                start_time=cumul_inc_start_time,
            )

        # Age-specific notifications
        for agegroup in AGEGROUP_STRATA:
            age_notification_pathways = []

            # First track all traced cases (regardless of clinical stratum)
            if contact_tracing_params:
                name = f"progress_tracedX{agegroup}"
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata={"tracing": "traced", "agegroup": agegroup},
                    save_results=False,
                )

            # Then track untraced cases that are passively detected (depending on clinical stratum)
            for clinical in NOTIFICATION_CLINICAL_STRATA:
                name = f"progress_untracedXagegroup_{agegroup}X{clinical}"
                dest_strata = {"clinical": clinical, "tracing": "untraced", "agegroup": agegroup} if \
                    contact_tracing_params else \
                    {"clinical": clinical, "agegroup": agegroup}
                age_notification_pathways.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata=dest_strata,
                    save_results=False,
                )
            self.model.request_aggregate_output(
                name=f"notificationsXagegroup_{agegroup}", sources=age_notification_pathways
            )

        # Split by child and adult
        paed_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[:3]]
        adult_notifications = [f"notificationsXagegroup_{agegroup}" for agegroup in AGEGROUP_STRATA[3:]]
        self.model.request_aggregate_output(name="notificationsXpaediatric", sources=paed_notifications)
        self.model.request_aggregate_output(name="notificationsXadult", sources=adult_notifications)

        self.request_extra_notifications(contact_tracing_params, cumul_inc_start_time)

    def request_extra_notifications(self, contact_tracing_params, cumul_inc_start_time):
        pass

    def request_progression(self):
        # Unstratified
        self.model.request_output_for_flow(name=PROGRESS, flow_name=PROGRESS)

        # Stratified by age group and clinical status
        request_double_stratified_output_for_flow(
            self.model, PROGRESS, AGEGROUP_STRATA, "agegroup", NOTIFICATION_CLINICAL_STRATA, "clinical"
        )

        self.request_extra_progression()

    def request_extra_progression(self):
        pass

    def request_cdr(self):
        self.model.request_computed_value_output("cdr")


class VicCovidOutputs(CovidOutputs):

    def request_extra_notifications(self, contact_tracing_params, cumul_inc_start_time):

        # Victorian cluster model outputs
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

        # Cluster-specific notifications
        for cluster in clusters:
            cluster_notification_sources = []

            # First track all traced cases (regardless of clinical stratum)
            if contact_tracing_params:
                name = f"progress_tracedX{cluster}"
                cluster_notification_sources.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata={"tracing": "traced", "cluster": cluster},
                    save_results=False,
                )

            # Then track untraced cases that are passively detected (depending on clinical stratum)
            for clinical in NOTIFICATION_CLINICAL_STRATA:
                name = f"progress_untracedXcluster_{cluster}X{clinical}"
                dest_strata = {"clinical": clinical, "cluster": cluster, "tracing": "untraced"} if \
                    contact_tracing_params else {"clinical": clinical, "cluster": cluster}
                cluster_notification_sources.append(name)
                self.model.request_output_for_flow(
                    name=name,
                    flow_name="progress",
                    dest_strata=dest_strata,
                    save_results=False,
                )

            self.model.request_aggregate_output(
                name=f"notificationsXcluster_{cluster}", sources=cluster_notification_sources
            )

    def request_extra_progression(self):

        # Stratified by cluster
        request_stratified_output_for_flow(
            self.model, PROGRESS,
            [region.replace("-", "_") for region in Region.VICTORIA_SUBREGIONS],
            "cluster", "progress_for_", filter_on="destination",
        )


def request_common_outputs(model: CompartmentalModel, is_region_vic):

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
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
        request_stratified_output_for_flow(model, INFECT_DEATH, clusters, "cluster", "source")
