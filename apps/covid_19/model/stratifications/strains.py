from summer import StrainStratification, Multiply

from apps.covid_19.constants import (
    DISEASE_COMPARTMENTS,
    Strain,
)


def get_strain_strat(params):
    strain_strat = StrainStratification(
        "strain",
        [Strain.WILD_TYPE, Strain.VARIANT_OF_CONCERN],
        DISEASE_COMPARTMENTS,

    )
    strain_strat.set_population_split(
        {Strain.WILD_TYPE: 1., Strain.VARIANT_OF_CONCERN: 0.}
    )
    strain_strat.add_flow_adjustments(
        "infection",
        {
            Strain.WILD_TYPE: None,
            Strain.VARIANT_OF_CONCERN: Multiply(params.voc_emergence.contact_rate_multiplier)
        }
    )
    return strain_strat
