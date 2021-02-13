from typing import List
import numpy as np

from summer2 import CompartmentalModel

from autumn.region import Region

from apps.covid_19.model.parameters import Parameters
from apps.covid_19.constants import Compartment, Clinical, NOTIFICATION_STRATA, COMPARTMENTS
from apps.covid_19.model.stratifications.agegroup import AGEGROUP_STRATA


def request_victorian_outputs(model: CompartmentalModel, params: Parameters):
    # Victorian cluster model outputs.
    clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
    # Track incidence of disease (transition from exposed to active) - overall and for each cluster
    model.request_output_for_flow(name="incidence", flow_name="incidence")
    for cluster in clusters:
        model.request_output_for_flow(
            name=f"incidence_for_cluster_{cluster}",
            flow_name="incidence",
            dest_strata={"cluster": cluster},
        )

    # Track progress of disease (transition from early to late active)
    # - overall, by cluster, by clinical stratum and by both
    model.request_output_for_flow(name="progress", flow_name="progress")
    for cluster in clusters:
        model.request_output_for_flow(
            name=f"progress_for_cluster_{cluster}",
            flow_name="progress",
            dest_strata={"cluster": cluster},
        )

    # Notifications.
    notification_sources = []
    for clinical in NOTIFICATION_STRATA:
        name = f"progressX{clinical}"
        notification_sources.append(name)
        model.request_output_for_flow(
            name=name,
            flow_name="progress",
            dest_strata={"clinical": clinical},
            save_results=False,
        )

    model.request_aggregate_output(name="notifications", sources=notification_sources)
    # Cluster-specific notifications.
    for cluster in clusters:
        cluster_notification_sources = []
        for clinical in NOTIFICATION_STRATA:
            name = f"progress_for_cluster_{cluster}X{clinical}"
            cluster_notification_sources.append(name)
            model.request_output_for_flow(
                name=name,
                flow_name="progress",
                dest_strata={"cluster": cluster, "clinical": clinical},
                save_results=False,
            )

        model.request_aggregate_output(
            name=f"notifications_for_cluster_{cluster}", sources=cluster_notification_sources
        )
        model.request_cumulative_output(
            name=f"accum_notifications_for_cluster_{cluster}",
            source=f"notifications_for_cluster_{cluster}",
        )

    # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
    model.request_output_for_flow(
        name="non_icu_admissions",
        flow_name="progress",
        source_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
        dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
    )
    for cluster in clusters:
        model.request_output_for_flow(
            name=f"non_icu_admissions_for_cluster_{cluster}",
            flow_name="progress",
            source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
            dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
        )

    # Track ICU admissions (transition from early to late active in ICU stratum)
    model.request_output_for_flow(
        name="icu_admissions",
        flow_name="progress",
        source_strata={"clinical": Clinical.ICU},
        dest_strata={"clinical": Clinical.ICU},
    )
    for cluster in clusters:
        model.request_output_for_flow(
            name=f"icu_admissions_for_cluster_{cluster}",
            flow_name="progress",
            source_strata={"clinical": Clinical.ICU, "cluster": cluster},
            dest_strata={"clinical": Clinical.ICU, "cluster": cluster},
        )
        model.request_cumulative_output(
            name=f"accum_icu_admissions_for_cluster_{cluster}",
            source=f"icu_admissions_for_cluster_{cluster}",
        )

    # Create hospitalisation functions as sum of hospital non-ICU and ICU
    model.request_aggregate_output(
        "hospital_admissions", sources=["icu_admissions", "non_icu_admissions"]
    )
    for cluster in clusters:
        model.request_aggregate_output(
            f"hospital_admissions_for_cluster_{cluster}",
            sources=[
                f"icu_admissions_for_cluster_{cluster}",
                f"non_icu_admissions_for_cluster_{cluster}",
            ],
        )
        model.request_cumulative_output(
            name=f"accum_hospital_admissions_for_cluster_{cluster}",
            source=f"hospital_admissions_for_cluster_{cluster}",
        )

    # Hospital occupancy
    # We count all ICU and hospital late active compartments and a proportion of early active ICU cases.
    compartment_periods = params.sojourn.compartment_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.0)
    proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period
    model.request_output_for_compartments(
        "_late_active_hospital",
        compartments=[Compartment.LATE_ACTIVE],
        strata={"clinical": Clinical.HOSPITAL_NON_ICU},
        save_results=False,
    )
    model.request_output_for_compartments(
        "icu_occupancy",
        compartments=[Compartment.LATE_ACTIVE],
        strata={"clinical": Clinical.ICU},
    )
    model.request_output_for_compartments(
        "_early_active_icu",
        compartments=[Compartment.EARLY_ACTIVE],
        strata={"clinical": Clinical.ICU},
        save_results=False,
    )
    model.request_function_output(
        name="_early_active_icu_proportion",
        func=lambda patients: patients * proportion_icu_patients_in_hospital,
        sources=["_early_active_icu"],
        save_results=False,
    )
    model.request_aggregate_output(
        name="hospital_occupancy",
        sources=[
            "_late_active_hospital",
            "icu_occupancy",
            "_early_active_icu_proportion",
        ],
    )
    for cluster in clusters:
        model.request_output_for_compartments(
            f"_late_active_hospital_for_cluster_{cluster}",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
            save_results=False,
        )
        model.request_output_for_compartments(
            f"icu_occupancy_for_cluster_{cluster}",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.ICU, "cluster": cluster},
        )
        model.request_output_for_compartments(
            f"_early_active_icu_for_cluster_{cluster}",
            compartments=[Compartment.EARLY_ACTIVE],
            strata={"clinical": Clinical.ICU, "cluster": cluster},
            save_results=False,
        )
        model.request_function_output(
            name=f"_early_active_icu_proportion_for_cluster_{cluster}",
            func=lambda patients: patients * proportion_icu_patients_in_hospital,
            sources=["_early_active_icu"],
            save_results=False,
        )
        model.request_aggregate_output(
            name=f"hospital_occupancy_for_cluster_{cluster}",
            sources=[
                f"_late_active_hospital_for_cluster_{cluster}",
                f"icu_occupancy_for_cluster_{cluster}",
                f"_early_active_icu_proportion_for_cluster_{cluster}",
            ],
        )

    # Infection deaths.
    model.request_output_for_flow(name="infection_deaths", flow_name="infect_death")
    model.request_cumulative_output(name="accum_deaths", source="infection_deaths")
    for cluster in clusters:
        model.request_output_for_flow(
            name=f"infection_deaths_for_cluster_{cluster}",
            flow_name="infect_death",
            source_strata={"cluster": cluster},
        )
        model.request_cumulative_output(
            name=f"accum_infection_deaths_for_cluster_{cluster}", source="infection_deaths"
        )

    # Proportion seropositive
    model.request_output_for_compartments(
        name="_total_population", compartments=COMPARTMENTS, save_results=False
    )
    model.request_output_for_compartments(
        name="_recovered", compartments=[Compartment.RECOVERED], save_results=False
    )
    model.request_function_output(
        name="proportion_seropositive",
        sources=["_recovered", "_total_population"],
        func=lambda recovered, total: recovered / total,
    )
    for agegroup in AGEGROUP_STRATA:
        total_name = f"_total_populationXagegroup_{agegroup}"
        recovered_name = f"_recoveredXagegroup_{agegroup}"
        model.request_output_for_compartments(
            name=total_name,
            compartments=COMPARTMENTS,
            strata={"agegroup": agegroup},
            save_results=False,
        )
        model.request_output_for_compartments(
            name=recovered_name,
            compartments=[Compartment.RECOVERED],
            strata={"agegroup": agegroup},
            save_results=False,
        )
        model.request_function_output(
            name=f"proportion_seropositiveXagegroup_{agegroup}",
            sources=[recovered_name, total_name],
            func=lambda recovered, total: recovered / total,
        )
    for cluster in clusters:
        total_name = f"_total_populationXcluster_{cluster}"
        recovered_name = f"_recoveredXcluster_{cluster}"
        model.request_output_for_compartments(
            name=total_name,
            compartments=COMPARTMENTS,
            strata={"cluster": cluster},
            save_results=False,
        )
        model.request_output_for_compartments(
            name=recovered_name,
            compartments=[Compartment.RECOVERED],
            strata={"cluster": cluster},
            save_results=False,
        )
        model.request_function_output(
            name=f"proportion_seropositiveX{cluster}",
            sources=[recovered_name, total_name],
            func=lambda recovered, total: recovered / total,
        )
