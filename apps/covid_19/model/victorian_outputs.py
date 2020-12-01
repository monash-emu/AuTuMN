from typing import Dict
import numpy as np

from summer.model import StratifiedModel
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)
from apps.covid_19.constants import Compartment as CompartmentType, ClinicalStratum
from autumn.constants import Region
from apps.covid_19.model.outputs import NOTIFICATION_STRATUM
from apps.covid_19.model.outputs import get_calculate_hospital_occupancy, calculate_icu_occupancy


CLUSTERS = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]


def add_victorian_derived_outputs(
        model: StratifiedModel,
        icu_early_period: float,
        hospital_early_period: float,
):

    # Track incidence of disease (transition from exposed to active) - overall and for each cluster
    incidence_conns = {
        "incidence": TransitionFlowOutput(
            source=CompartmentType.LATE_EXPOSED,
            dest=CompartmentType.EARLY_ACTIVE,
            source_strata={},
            dest_strata={},
        )
    }
    for cluster in CLUSTERS:
        output_key = f"incidence_for_cluster_{cluster}"
        incidence_conns[output_key] = TransitionFlowOutput(
            source=CompartmentType.LATE_EXPOSED,
            dest=CompartmentType.EARLY_ACTIVE,
            source_strata={},
            dest_strata={"cluster": cluster},
        )
    model.add_flow_derived_outputs(incidence_conns)

    # Track progress of disease (transition from early to late active)
    # - overall, by cluster, by clinical stratum and by both
    progress_connections = {
        "progress": TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={},
        )
    }
    for cluster in CLUSTERS:
        progress_connections[f"progress_for_cluster_{cluster}"] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={"cluster": cluster},
        )
    for clinical_stratum in NOTIFICATION_STRATUM:
        progress_connections[f"progressX{clinical_stratum}"] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={"clinical": clinical_stratum},
        )
        for cluster in CLUSTERS:
            output_key = f"progress_for_cluster_{cluster}X{clinical_stratum}"
            progress_connections[output_key] = TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={},
                dest_strata={"cluster": cluster, "clinical": clinical_stratum},
            )
    model.add_flow_derived_outputs(progress_connections)

    # Notification aggregation functions
    model.add_function_derived_outputs(
        {f"notifications_for_cluster_{cluster}":
             build_cluster_notification_func(cluster)
         for cluster in CLUSTERS}
    )
    model.add_function_derived_outputs(
        {"notifications": total_notification_func}
    )

    # Track non-ICU hospital admissions (transition from early to late active in hospital, non-ICU stratum)
    non_icu_admit_connections = {
        f"non_icu_admissions":
            TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={"clinical": ClinicalStratum.HOSPITAL_NON_ICU},
                dest_strata={"clinical": ClinicalStratum.HOSPITAL_NON_ICU}
            )
    }
    for cluster in CLUSTERS:
        non_icu_admit_connections[f"non_icu_admissions_for_cluster_{cluster}"] = \
            TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={
                    "cluster": cluster,
                    "clinical": ClinicalStratum.HOSPITAL_NON_ICU,
                },
                dest_strata={
                    "cluster": cluster,
                    "clinical": ClinicalStratum.HOSPITAL_NON_ICU,
                }
            )
    model.add_flow_derived_outputs(non_icu_admit_connections)

    # Track ICU admissions (transition from early to late active in ICU stratum)
    icu_admit_connections = {
        f"icu_admissions":
            TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={"clinical": ClinicalStratum.ICU},
                dest_strata={"clinical": ClinicalStratum.ICU}
            )
    }
    for cluster in CLUSTERS:
        icu_admit_connections[f"icu_admissions_for_cluster_{cluster}"] = \
            TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={"cluster": cluster, "clinical": ClinicalStratum.ICU},
                dest_strata={"cluster": cluster, "clinical": ClinicalStratum.ICU}
            )
    model.add_flow_derived_outputs(icu_admit_connections)

    # Create hospitalisation functions as sum of hospital non-ICU and ICU
    model.add_function_derived_outputs(
        {f"hospital_admissions": get_hospitalisation_func()}
    )
    model.add_function_derived_outputs(
        {f"hospital_admissions_for_cluster_{cluster}":
             get_cluster_hospitalisation_func(cluster)
         for cluster in CLUSTERS}
    )

    # Hospital occupancy
    model.add_function_derived_outputs(
        {"hospital_occupancy": get_calculate_hospital_occupancy(icu_early_period, hospital_early_period)}
    )
    model.add_function_derived_outputs(
        {f"hospital_occupancy_for_cluster_{cluster}":
             get_calculate_cluster_hospital_occupancy(icu_early_period, hospital_early_period, cluster)
         for cluster in CLUSTERS}
    )

    # ICU occupancy
    model.add_function_derived_outputs(
        {"icu_occupancy": calculate_icu_occupancy}
    )
    model.add_function_derived_outputs(
        {f"icu_occupancy_for_cluster_{cluster}":
             get_calculate_cluster_icu_occupancy(cluster) for
         cluster in CLUSTERS}
    )

    # Track infection deaths - overall and for each cluster
    inf_death_conns = {
        "infection_deaths":
            InfectionDeathFlowOutput(
                source=CompartmentType.LATE_ACTIVE,
                source_strata={}
            )
    }
    for cluster in CLUSTERS:
        output_key = f"infection_deaths_for_cluster_{cluster}"
        inf_death_conns[output_key] = InfectionDeathFlowOutput(
            source=CompartmentType.LATE_ACTIVE,
            source_strata={"cluster": cluster},
        )
    model.add_flow_derived_outputs(inf_death_conns)


def build_cluster_notification_func(cluster: str):
    def notification_func(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        count = 0.0
        for clinical_stratum in NOTIFICATION_STRATUM:
            output_key = f"progress_for_cluster_{cluster}X{clinical_stratum}"
            count += derived_outputs[output_key][time_idx]
        return count

    return notification_func


def total_notification_func(
    time_idx: int,
    model: StratifiedModel,
    compartment_values: np.ndarray,
    derived_outputs: Dict[str, np.ndarray],
):
    count = 0.0
    for cluster in CLUSTERS:
        output_key = f"notifications_for_cluster_{cluster}"
        count += derived_outputs[output_key][time_idx]
    return count


def get_hospitalisation_func():
    def hospitalisation_func(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        count = 0.0
        count += derived_outputs[f"non_icu_admissions"][time_idx]
        count += derived_outputs[f"icu_admissions"][time_idx]
        return count

    return hospitalisation_func


def get_cluster_hospitalisation_func(cluster: str):
    def hospitalisation_func(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        count = 0.0
        count += derived_outputs[f"non_icu_admissions_for_cluster_{cluster}"][time_idx]
        count += derived_outputs[f"icu_admissions_for_cluster_{cluster}"][time_idx]
        return count

    return hospitalisation_func


def get_calculate_cluster_hospital_occupancy(icu_early_period, hospital_early_period, cluster):
    def calculate_hospital_occupancy(
        time_idx: int,
        model: StratifiedModel,
        compartment_values: np.ndarray,
        derived_outputs: Dict[str, np.ndarray],
    ):
        hospital_prev = 0.0
        period_icu_patients_in_hospital = max(
            icu_early_period - hospital_early_period,
            0.0,
        )
        proportion_icu_patients_in_hospital = period_icu_patients_in_hospital / icu_early_period
        for i, comp in enumerate(model.compartment_names):
            is_late_active = comp.has_name(CompartmentType.LATE_ACTIVE)
            is_early_active = comp.has_name(CompartmentType.EARLY_ACTIVE)
            is_icu = comp.has_stratum("clinical", ClinicalStratum.ICU)
            is_hospital_non_icu = comp.has_stratum("clinical", ClinicalStratum.HOSPITAL_NON_ICU)
            is_cluster = comp.has_stratum("cluster", cluster)
            if is_late_active and (is_icu or is_hospital_non_icu) and is_cluster:
                # Both ICU and hospital late active compartments
                hospital_prev += compartment_values[i]
            elif is_early_active and is_icu and is_cluster:
                # A proportion of the early active ICU compartment
                hospital_prev += compartment_values[i] * proportion_icu_patients_in_hospital

        return hospital_prev

    return calculate_hospital_occupancy


def get_calculate_cluster_icu_occupancy(cluster):
    def calculate_cluster_icu_occupancy(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        icu_prev = 0
        for i, comp in enumerate(model.compartment_names):
            is_late_active = comp.has_name(CompartmentType.LATE_ACTIVE)
            is_icu = comp.has_stratum("clinical", ClinicalStratum.ICU)
            is_cluster = comp.has_stratum("cluster", cluster)
            if is_late_active and is_icu and is_cluster:
                icu_prev += compartment_values[i]
        return icu_prev

    return calculate_cluster_icu_occupancy
