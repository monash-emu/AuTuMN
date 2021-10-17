from .common import CovidOutputs
from autumn.models.covid_19.constants import (
    INFECT_DEATH, Compartment, PROGRESS, NOTIFICATION_CLINICAL_STRATA, COMPARTMENTS
)
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.settings import Region
from autumn.models.covid_19.constants import Clinical


class VicCovidOutputs(CovidOutputs):

    def __init__(self, model, COMPARTMENTS):
        self.model = model
        self.model.request_output_for_compartments(
            name="_total_population", compartments=COMPARTMENTS, save_results=False
        )
        self.clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

    def request_extra_notifications(self, contact_tracing_params, cumul_inc_start_time):

        # Cluster-specific notifications
        for cluster in self.clusters:
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
        self.request_stratified_output_for_flow(
            PROGRESS,
            [region.replace("-", "_") for region in Region.VICTORIA_SUBREGIONS],
            "cluster", "progress_for_", filter_on="destination",
        )

    def request_extra_deaths(self):
        self.request_stratified_output_for_flow(INFECT_DEATH, self.clusters, "cluster", "source")

    def request_extra_admissions(self):

        for cluster in self.clusters:
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

        for cluster in self.clusters:
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

    def request_extra_recovered(self):

        for cluster in self.clusters:
            total_name = f"_total_populationXcluster_{cluster}"
            recovered_name = f"_recoveredXcluster_{cluster}"
            self.model.request_output_for_compartments(
                name=total_name,
                compartments=COMPARTMENTS,
                strata={"cluster": cluster},
                save_results=False,
            )
            self.model.request_output_for_compartments(
                name=recovered_name,
                compartments=[Compartment.RECOVERED],
                strata={"cluster": cluster},
                save_results=False,
            )
            self.model.request_function_output(
                name=f"proportion_seropositiveXcluster_{cluster}",
                sources=[recovered_name, total_name],
                func=lambda recovered, total: recovered / total,
            )
