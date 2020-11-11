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


def add_victorian_derived_outputs(
    model: StratifiedModel, icu_early_period: float, hospital_early_period: float
):

    # Track infection deaths
    inf_death_conns = {}
    inf_death_conns["infection_deaths"] = InfectionDeathFlowOutput(
        source=CompartmentType.LATE_ACTIVE, source_strata={}
    )
    for cluster in CLUSTERS:
        output_key = f"infection_deaths_for_cluster_{cluster}"
        inf_death_conns[output_key] = InfectionDeathFlowOutput(
            source=CompartmentType.LATE_ACTIVE,
            source_strata={"cluster": cluster},
        )

    model.add_flow_derived_outputs(inf_death_conns)

    # Track incidence of disease: transition from exposed to active
    incidence_conns = {}
    incidence_conns["incidence"] = TransitionFlowOutput(
        source=CompartmentType.LATE_EXPOSED,
        dest=CompartmentType.EARLY_ACTIVE,
        source_strata={},
        dest_strata={},
    )
    for cluster in CLUSTERS:
        output_key = f"incidence_for_cluster_{cluster}"
        incidence_conns[output_key] = TransitionFlowOutput(
            source=CompartmentType.LATE_EXPOSED,
            dest=CompartmentType.EARLY_ACTIVE,
            source_strata={},
            dest_strata={"cluster": cluster},
        )

    model.add_flow_derived_outputs(incidence_conns)

    # Track progress of disease: transition from early to late active
    progress_conns = {}
    progress_conns["progress"] = TransitionFlowOutput(
        source=CompartmentType.EARLY_ACTIVE,
        dest=CompartmentType.LATE_ACTIVE,
        source_strata={},
        dest_strata={},
    )
    for cluster in CLUSTERS:
        output_key = f"progress_for_cluster_{cluster}"
        progress_conns[output_key] = TransitionFlowOutput(
            source=CompartmentType.EARLY_ACTIVE,
            dest=CompartmentType.LATE_ACTIVE,
            source_strata={},
            dest_strata={"cluster": cluster},
        )

    model.add_flow_derived_outputs(progress_conns)

    # Track notifications: when we know someone has progressed from early to late active
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

    # Notification aggregation functions
    model.add_function_derived_outputs(
        {f"notifications_for_cluster_{c}": build_cluster_notification_func(c) for c in CLUSTERS}
    )
    model.add_function_derived_outputs({"notifications": total_notification_func})

    # TODO?
    # hospital_occupancy
    # icu_occupancy
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