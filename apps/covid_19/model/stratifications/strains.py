from summer import StrainStratification

from apps.covid_19.constants import (
    INFECTIOUS_COMPARTMENTS,
    Compartment,
)


def get_strain_strat():
    strat = StrainStratification(
        "strain",
        ["wild", "voc"],
        INFECTIOUS_COMPARTMENTS + [Compartment.EARLY_EXPOSED]
    )
    strat.set_population_split(
        {"wild": 1., "voc": 0.}
    )
    return strat
