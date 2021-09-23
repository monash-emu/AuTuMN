from summer import CompartmentalModel

from autumn.models.covid_19.constants import Compartment
from autumn.settings import Region
from autumn.models.covid_19.constants import Clinical
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def request_healthcare_outputs(model: CompartmentalModel, sojourn_periods, is_region_vic: bool):

    # Clusters to cycle over for Vic model if needed
    clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS] if is_region_vic else ()

    """
    New admissions to hospital and ICU (previously had the string "new" and the front)
    """

    # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
    model.request_output_for_flow(
        name="non_icu_admissions",
        flow_name="progress",
        source_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
        dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU},
        save_results=False,
    )

    # Track ICU admissions (transition from early to late active in ICU stratum)
    model.request_output_for_flow(
        name="icu_admissions",
        flow_name="progress",
        source_strata={"clinical": Clinical.ICU},
        dest_strata={"clinical": Clinical.ICU},
    )

    # Create hospitalisation functions as sum of hospital non-ICU and ICU
    model.request_aggregate_output(
        "hospital_admissions",
        sources=["icu_admissions", "non_icu_admissions"]
    )

    for cluster in clusters:
        cluster_extension = f"Xcluster_{cluster}"
        model.request_output_for_flow(
            name=f"non_icu_admissions{cluster_extension}",
            flow_name="progress",
            source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
            dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
        )
        model.request_output_for_flow(
            name=f"icu_admissions{cluster_extension}",
            flow_name="progress",
            source_strata={"clinical": Clinical.ICU, "cluster": cluster},
            dest_strata={"clinical": Clinical.ICU, "cluster": cluster},
        )
        model.request_aggregate_output(
            f"hospital_admissions{cluster_extension}",
            sources=[
                f"icu_admissions{cluster_extension}",
                f"non_icu_admissions{cluster_extension}",
            ],
        )
        for agegroup in AGEGROUP_STRATA:
            cluster_age_extension = f"Xcluster_{cluster}Xagegroup_{agegroup}"
            model.request_output_for_flow(
                name=f"non_icu_admissions{cluster_age_extension}",
                flow_name="progress",
                source_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster, "agegroup": agegroup},
                dest_strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster, "agegroup": agegroup},
            )
            model.request_output_for_flow(
                name=f"icu_admissions{cluster_age_extension}",
                flow_name="progress",
                source_strata={"clinical": Clinical.ICU, "cluster": cluster},
                dest_strata={"clinical": Clinical.ICU, "cluster": cluster},
            )
            model.request_aggregate_output(
                f"hospital_admissions{cluster_age_extension}",
                sources=[
                    f"icu_admissions{cluster_age_extension}",
                    f"non_icu_admissions{cluster_age_extension}",
                ],
            )

    """
    Healthcare occupancy (hospital and ICU)
    """

    # Hospital occupancy is represented as all ICU, all hospital late active, and some early active ICU cases
    compartment_periods = sojourn_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
    proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period

    # Unstratified calculations
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
        cluster_extension = f"Xcluster_{cluster}"
        model.request_output_for_compartments(
            f"_late_active_hospital{cluster_extension}",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
            save_results=False,
        )
        model.request_output_for_compartments(
            f"icu_occupancy{cluster_extension}",
            compartments=[Compartment.LATE_ACTIVE],
            strata={"clinical": Clinical.ICU, "cluster": cluster},
        )
        model.request_output_for_compartments(
            f"_early_active_icu{cluster_extension}",
            compartments=[Compartment.EARLY_ACTIVE],
            strata={"clinical": Clinical.ICU, "cluster": cluster},
            save_results=False,
        )
        model.request_function_output(
            name=f"_early_active_icu_proportion{cluster_extension}",
            func=lambda patients: patients * proportion_icu_patients_in_hospital,
            sources=["_early_active_icu"],
            save_results=False,
        )
        model.request_aggregate_output(
            name=f"hospital_occupancy{cluster_extension}",
            sources=[
                f"_late_active_hospital{cluster_extension}",
                f"icu_occupancy{cluster_extension}",
                f"_early_active_icu_proportion{cluster_extension}",
            ],
        )
        for agegroup in AGEGROUP_STRATA:
            cluster_age_extension = f"Xcluster_{cluster}Xagegroup_{agegroup}"
            model.request_output_for_compartments(
                f"_late_active_hospital{cluster_age_extension}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
                save_results=False,
            )
            model.request_output_for_compartments(
                f"icu_occupancy{cluster_age_extension}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
            )
            model.request_output_for_compartments(
                f"_early_active_icu{cluster_age_extension}",
                compartments=[Compartment.EARLY_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
                save_results=False,
            )
            model.request_function_output(
                name=f"_early_active_icu_proportion{cluster_age_extension}",
                func=lambda patients: patients * proportion_icu_patients_in_hospital,
                sources=["_early_active_icu"],
                save_results=False,
            )
            model.request_aggregate_output(
                name=f"hospital_occupancy{cluster_age_extension}",
                sources=[
                    f"_late_active_hospital{cluster_age_extension}",
                    f"icu_occupancy{cluster_age_extension}",
                    f"_early_active_icu_proportion{cluster_age_extension}",
                ],
            )
