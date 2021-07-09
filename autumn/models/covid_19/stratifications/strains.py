from summer import StrainStratification, Multiply

from autumn.models.covid_19.constants import (
    DISEASE_COMPARTMENTS,
    Strain,
)


def get_strain_strat(contact_rate_multiplier):
    """
    Stratify the model by strain, with two strata, being wild or "ancestral" virus type and the single variant of concern ("VoC")

    """

    # Stratify model.
    strain_strat = StrainStratification(
        "strain",
        [Strain.WILD_TYPE, Strain.VARIANT_OF_CONCERN],
        DISEASE_COMPARTMENTS,
    )

    # Assign all population to the wild type.
    strain_strat.set_population_split({Strain.WILD_TYPE: 1.0, Strain.VARIANT_OF_CONCERN: 0.0})

    # Make the VoC stratum more transmissible.
    strain_strat.add_flow_adjustments(
        "infection",
        {Strain.WILD_TYPE: None, Strain.VARIANT_OF_CONCERN: Multiply(contact_rate_multiplier)},
    )
    return strain_strat

def get_strain_strat_dual_voc(contact_rate_multiplier,contact_rate_multiplier_second_VoC):
    """
    Stratify the model by strain, with three strata, being wild or "ancestral" virus type, variant of concern ("VoC") and another highly transmissible variant of concern ("VoC").

    """

    # Stratify model.
    strain_strat = StrainStratification(
        "strain",
        [Strain.WILD_TYPE, Strain.VARIANT_OF_CONCERN, Strain.ADDITIONAL_VARIANT_OF_CONCERN],
        DISEASE_COMPARTMENTS,
    )

    # Assign all population to the wild type.
    strain_strat.set_population_split({Strain.WILD_TYPE: 1.0, Strain.VARIANT_OF_CONCERN: 0.0,
                                      Strain.ADDITIONAL_VARIANT_OF_CONCERN: 0.0},)

    # Make the VoC stratum more transmissible.
    strain_strat.add_flow_adjustments(
        "infection",
        {Strain.WILD_TYPE: None, Strain.VARIANT_OF_CONCERN: Multiply(contact_rate_multiplier),
         Strain.ADDITIONAL_VARIANT_OF_CONCERN: Multiply(contact_rate_multiplier_second_VoC)},
    )
    return strain_strat