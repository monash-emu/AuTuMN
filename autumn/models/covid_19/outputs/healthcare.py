from summer import CompartmentalModel

from autumn.models.covid_19.constants import Compartment, Clinical
from autumn.settings import Region


def request_healthcare_outputs(model: CompartmentalModel, sojourn_periods, is_region_vic: bool):
    """
    Healthcare occupancy
    """

    clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]

    # Hospital occupancy represented as all ICU, all hospital late active, and some early active ICU cases
    compartment_periods = sojourn_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
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

    #
    compartment_periods = sojourn_periods
    icu_early_period = compartment_periods["icu_early"]
    hospital_early_period = compartment_periods["hospital_early"]
    period_icu_patients_in_hospital = max(icu_early_period - hospital_early_period, 0.)
    proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period

    if is_region_vic:
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
