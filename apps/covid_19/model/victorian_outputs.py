from typing import Dict
import numpy as np

from summer.model.strat_model import StratifiedModel
from summer.model import StratifiedModel
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)
from apps.covid_19.constants import Compartment as CompartmentType, ClinicalStratum
from autumn.constants import Region
from apps.covid_19.model.outputs import NOTIFICATION_STRATUM


CLUSTERS = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]


def add_victorian_derived_outputs(model: StratifiedModel):

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

    # Track progress of disease (transition from early to late active) - overall and for each cluster
    progress_conns = {
        "progress": TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={},
        )
    }
    for cluster in CLUSTERS:
        progress_conns[f"progress_for_cluster_{cluster}"] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={"cluster": cluster},
        )
    model.add_flow_derived_outputs(progress_conns)

    # Track notifications (transition from early to late active) - by clinical stratum and by both clinical stratum and
    # cluster
    # This is defined as the person who transitions ending up in isolation or hospital.
    # We aggregate over each strata in a function later on.
    notification_conns = {}
    for clinical_stratum in NOTIFICATION_STRATUM:
        notification_conns[f"notificationsX{clinical_stratum}"] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={"clinical": clinical_stratum},
        )
        for cluster in CLUSTERS:
            output_key = f"notifications_for_cluster_{cluster}X{clinical_stratum}"
            notification_conns[output_key] = TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={},
                dest_strata={"cluster": cluster, "clinical": clinical_stratum},
            )
    model.add_flow_derived_outputs(notification_conns)

    hospital_conns = {}
    for cluster in CLUSTERS:
        hospital_conns[f"non_icu_admissions_for_cluster_{cluster}"] = \
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
    model.add_flow_derived_outputs(hospital_conns)

    icu_conns = {
        f"icu_admissions":
            TransitionFlowOutput(
                source=CompartmentType.EARLY_ACTIVE,
                dest=CompartmentType.LATE_ACTIVE,
                source_strata={"clinical": ClinicalStratum.ICU},
                dest_strata={"clinical": ClinicalStratum.ICU}
            )
    }
    for cluster in CLUSTERS:
        output_key = f"icu_admissions_for_cluster_{cluster}"
        icu_conns[output_key] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={
                "cluster": cluster,
                "clinical": ClinicalStratum.ICU,
            },
            dest_strata={
                "cluster": cluster,
                "clinical": ClinicalStratum.ICU,
            }
        )
    model.add_flow_derived_outputs(icu_conns)

    model.add_function_derived_outputs(
        {f"hospital_admissions_for_cluster_{c}": build_cluster_hospitalisation_func(c) for c in CLUSTERS}
    )

    # Notification aggregation functions
    model.add_function_derived_outputs(
        {f"notifications_for_cluster_{c}": build_cluster_notification_func(c) for c in CLUSTERS}
    )
    model.add_function_derived_outputs({"notifications": total_notification_func})

    # icu_admissions
    # hospital_admissions


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


def build_cluster_notification_func(cluster: str):
    def notification_func(
            time_idx: int,
            model: StratifiedModel,
            compartment_values: np.ndarray,
            derived_outputs: Dict[str, np.ndarray],
    ):
        count = 0.0
        for clinical_stratum in NOTIFICATION_STRATUM:
            output_key = f"notifications_for_cluster_{cluster}X{clinical_stratum}"
            count += derived_outputs[output_key][time_idx]

        return count

    return notification_func


def build_cluster_hospitalisation_func(cluster: str):
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
