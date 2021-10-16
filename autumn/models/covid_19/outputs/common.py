from autumn.models.covid_19.constants import (
    INFECT_DEATH, INFECTION, Compartment, PROGRESS, NOTIFICATIONS, NOTIFICATION_CLINICAL_STRATA, INCIDENCE,
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.clinical import CLINICAL_STRATA
from autumn.settings import Region
from autumn.models.covid_19.constants import Clinical

""" *** I have no idea why I can't kill the following function *** """


def cant_kill_function():
    pass


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
    """
    This class is not specific to Covid, so should be moved out of this file - but not sure whether to move it to
    somewhere in autumn or in summer.
    """

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

    def request_deaths(self):

        # Unstratified
        self.model.request_output_for_flow(name="infection_deaths", flow_name=INFECT_DEATH)
        self.model.request_cumulative_output(name="accum_deaths", source="infection_deaths")

        # Stratified by age
        self.request_stratified_output_for_flow(
            INFECT_DEATH, AGEGROUP_STRATA, "agegroup", name_stem="infection_deaths", filter_on="source"
        )
        for agegroup in AGEGROUP_STRATA:
            self.model.request_cumulative_output(
                name=f"accum_deathsXagegroup_{agegroup}",
                source=f"infection_deathsXagegroup_{agegroup}",
            )

        # Stratified by age and clinical stratum
        request_double_stratified_output_for_flow(
            self.model, INFECT_DEATH, AGEGROUP_STRATA, "agegroup",
            CLINICAL_STRATA, "clinical", name_stem="infection_deaths", filter_on="source"
        )

        self.request_extra_deaths()

    def request_extra_deaths(self):
        pass

    def request_admissions(self):

        # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
        self.model.request_output_for_flow(
            name="non_icu_admissions",
            flow_name="progress",
            source_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )

        # Track ICU admissions (transition from early to late active in ICU stratum)
        self.model.request_output_for_flow(
            name="icu_admissions",
            flow_name="progress",
            source_strata={"clinical": Clinical.ICU},
            dest_strata={"clinical": Clinical.ICU},
        )

        # Create hospitalisation functions as sum of hospital non-ICU and ICU
        self.model.request_aggregate_output(
            "hospital_admissions",
            sources=["icu_admissions", "non_icu_admissions"]
        )

        self.request_extra_admissions()

    def request_extra_admissions(self):
        pass

    def request_occupancy(self, sojourn_periods):

        # Hospital occupancy is represented as all ICU, all hospital late active, and some early active ICU cases
        compartment_periods = sojourn_periods
        icu_early_period = compartment_periods["icu_early"]
        hospital_early_period = compartment_periods["hospital_early"]
        period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
        self.proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period

        # Unstratified calculations
        self.model.request_output_for_compartments(
            "_late_active_hospital",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.HOSPITAL_NON_ICU},
            save_results=False,
        )
        self.model.request_output_for_compartments(
            "icu_occupancy",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.ICU},
        )
        self.model.request_output_for_compartments(
            "_early_active_icu",
            compartments=[Compartment.EARLY_ACTIVE],
            strata={"clinical": Clinical.ICU},
            save_results=False,
        )
        self.model.request_function_output(
            name="_early_active_icu_proportion",
            func=lambda patients: patients * self.proportion_icu_patients_in_hospital,
            sources=["_early_active_icu"],
            save_results=False,
        )
        self.model.request_aggregate_output(
            name="hospital_occupancy",
            sources=[
                "_late_active_hospital",
                "icu_occupancy",
                "_early_active_icu_proportion",
            ],
        )

        self.request_extra_occupancy()

    def request_extra_occupancy(self):
        pass


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

    def request_extra_deaths(self):
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
        self.request_stratified_output_for_flow(INFECT_DEATH, clusters, "cluster", "source")

    def request_extra_admissions(self):

        # Clusters to cycle over for Vic model if needed
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

        for cluster in clusters:
            cluster_extension = f"Xcluster_{cluster}"
            self.model.request_output_for_flow(
                name=f"non_icu_admissions{cluster_extension}",
                flow_name="progress",
                source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
                dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
                save_results=False,
            )
            self.model.request_output_for_flow(
                name=f"icu_admissions{cluster_extension}",
                flow_name="progress",
                source_strata={"clinical": Clinical.ICU, "cluster": cluster},
                dest_strata={"clinical": Clinical.ICU, "cluster": cluster},
            )
            self.model.request_aggregate_output(
                f"hospital_admissions{cluster_extension}",
                sources=[
                    f"icu_admissions{cluster_extension}",
                    f"non_icu_admissions{cluster_extension}",
                ],
            )
            for agegroup in AGEGROUP_STRATA:
                cluster_age_extension = f"Xcluster_{cluster}Xagegroup_{agegroup}"
                self.model.request_output_for_flow(
                    name=f"non_icu_admissions{cluster_age_extension}",
                    flow_name="progress",
                    source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster, "agegroup": agegroup},
                    dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster, "agegroup": agegroup},
                    save_results=False,
                )
                self.model.request_output_for_flow(
                    name=f"icu_admissions{cluster_age_extension}",
                    flow_name="progress",
                    source_strata={"clinical": Clinical.ICU, "cluster": cluster},
                    dest_strata={"clinical": Clinical.ICU, "cluster": cluster},
                )
                self.model.request_aggregate_output(
                    f"hospital_admissions{cluster_age_extension}",
                    sources=[
                        f"icu_admissions{cluster_age_extension}",
                        f"non_icu_admissions{cluster_age_extension}",
                    ],
                )

    def request_extra_occupancy(self):
        """
        Healthcare occupancy (hospital and ICU)
        """
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

        for cluster in clusters:
            cluster_extension = f"Xcluster_{cluster}"
            self.model.request_output_for_compartments(
                f"_late_active_hospital{cluster_extension}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
                save_results=False,
            )
            self.model.request_output_for_compartments(
                f"icu_occupancy{cluster_extension}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
            )
            self.model.request_output_for_compartments(
                f"_early_active_icu{cluster_extension}",
                compartments=[Compartment.EARLY_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
                save_results=False,
            )
            self.model.request_function_output(
                name=f"_early_active_icu_proportion{cluster_extension}",
                func=lambda patients: patients * self.proportion_icu_patients_in_hospital,
                sources=[f"_early_active_icu{cluster_extension}"],
                save_results=False,
            )
            self.model.request_aggregate_output(
                name=f"hospital_occupancy{cluster_extension}",
                sources=[
                    f"_late_active_hospital{cluster_extension}",
                    f"icu_occupancy{cluster_extension}",
                    f"_early_active_icu_proportion{cluster_extension}",
                ],
            )
            for agegroup in AGEGROUP_STRATA:
                cluster_age_extension = f"Xcluster_{cluster}Xagegroup_{agegroup}"
                self.model.request_output_for_compartments(
                    f"_late_active_hospital{cluster_age_extension}",
                    compartments=[Compartment.LATE_ACTIVE],
                    strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster, "agegroup": agegroup},
                    save_results=False,
                )
                self.model.request_output_for_compartments(
                    f"icu_occupancy{cluster_age_extension}",
                    compartments=[Compartment.LATE_ACTIVE],
                    strata={"clinical": Clinical.ICU, "cluster": cluster, "agegroup": agegroup},
                )
                self.model.request_output_for_compartments(
                    f"_early_active_icu{cluster_age_extension}",
                    compartments=[Compartment.EARLY_ACTIVE],
                    strata={"clinical": Clinical.ICU, "cluster": cluster, "agegroup": agegroup},
                    save_results=False,
                )
                self.model.request_function_output(
                    name=f"_early_active_icu_proportion{cluster_age_extension}",
                    func=lambda patients: patients * self.proportion_icu_patients_in_hospital,
                    sources=[f"_early_active_icu{cluster_age_extension}"],
                    save_results=False,
                )
                self.model.request_aggregate_output(
                    name=f"hospital_occupancy{cluster_age_extension}",
                    sources=[
                        f"_late_active_hospital{cluster_age_extension}",
                        f"icu_occupancy{cluster_age_extension}",
                        f"_early_active_icu_proportion{cluster_age_extension}",
                    ],
                )

