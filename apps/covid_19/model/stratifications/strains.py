from summer import StrainStratification, Multiply

from apps.covid_19.constants import (
    DISEASE_COMPARTMENTS,
    Strain,
)


def get_strain_strat(contact_rate_multiplier):
    """
    Stratify the model by strain, with two strata, being variant of concern ("VoC") and wild or "ancestral" virus type.

    """

    # Stratify model.
    strain_strat = StrainStratification(
        "strain",
        [Strain.WILD_TYPE, Strain.VARIANT_OF_CONCERN],
        DISEASE_COMPARTMENTS,
    )

    # Assign all population to the wild type.
    strain_strat.set_population_split(
        {Strain.WILD_TYPE: 1., Strain.VARIANT_OF_CONCERN: 0.}
    )

    # Make the VoC stratum more transmissible.
    strain_strat.add_flow_adjustments(
        "infection",
        {
            Strain.WILD_TYPE: None,
            Strain.VARIANT_OF_CONCERN: Multiply(contact_rate_multiplier)
        }
    )
    return strain_strat
