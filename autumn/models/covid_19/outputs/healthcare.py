from summer import CompartmentalModel

from autumn.models.covid_19.constants import Compartment
from autumn.settings import Region
from autumn.models.covid_19.constants import Clinical, PROGRESS
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def request_healthcare_outputs(model: CompartmentalModel, sojourn_periods, is_region_vic: bool):

    """
    New admissions
    """

    hospital_sources, icu_sources = [], []
    for agegroup in AGEGROUP_STRATA:
        hospital_sources += [
            f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}",
            f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}",
        ]
        icu_sources.append(
            f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}"
        )
    model.request_aggregate_output(name="new_hospital_admissions", sources=hospital_sources)
    model.request_aggregate_output(name="new_icu_admissions", sources=icu_sources)

    if is_region_vic:

        # ********

        # Clusters to cycle over for Vic model if needed
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

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

    if is_region_vic:

        for cluster in clusters:
            model.request_output_for_compartments(
                f"_late_active_hospitalXcluster_{cluster}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.HOSPITAL_NON_ICU, "cluster": cluster},
                save_results=False,
            )
            model.request_output_for_compartments(
                f"icu_occupancyXcluster_{cluster}",
                compartments=[Compartment.LATE_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
            )
            model.request_output_for_compartments(
                f"_early_active_icuXcluster_{cluster}",
                compartments=[Compartment.EARLY_ACTIVE],
                strata={"clinical": Clinical.ICU, "cluster": cluster},
                save_results=False,
            )
            model.request_function_output(
                name=f"_early_active_icu_proportionXcluster_{cluster}",
                func=lambda patients: patients * proportion_icu_patients_in_hospital,
                sources=["_early_active_icu"],
                save_results=False,
            )
            model.request_aggregate_output(
                name=f"hospital_occupancyXcluster_{cluster}",
                sources=[
                    f"_late_active_hospitalXcluster_{cluster}",
                    f"icu_occupancyXcluster_{cluster}",
                    f"_early_active_icu_proportionXcluster_{cluster}",
                ],
            )
