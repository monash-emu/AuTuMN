from summer.model.strat_model import StratifiedModel
from summer.model import StratifiedModel
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)
from apps.covid_19.constants import Compartment as CompartmentType, ClinicalStratum
from autumn.constants import Region


CLUSTERS = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]


def add_victorian_derived_outputs(model: StratifiedModel):

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

    # Track incidence
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

    # Track progress
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

    # notifications
    # hospital_occupancy
    # icu_occupancy
    # icu_admissions
    # hospital_admissions
