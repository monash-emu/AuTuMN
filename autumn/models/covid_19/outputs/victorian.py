from summer import CompartmentalModel

from autumn.models.covid_19.constants import NOTIFICATION_CLINICAL_STRATA, Clinical
from autumn.models.covid_19.parameters import Parameters
from autumn.settings import Region


def request_victorian_outputs(model: CompartmentalModel, params: Parameters):

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

