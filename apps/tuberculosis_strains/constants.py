class Compartment:
    """
    A tuberculosis model compartment.
    """

    SUSCEPTIBLE = "susceptible"
    EARLY_LATENT = "early_latent"
    LATE_LATENT = "late_latent"
    INFECTIOUS = "infectious"
    DETECTED = "detected"
    ON_TREATMENT = "on_treatment"
    RECOVERED = "recovered"


class OrganStratum:
    """
    A classification of TB active disease, based on the organ affected and smear-status
    """

    SMEAR_POSITIVE = "smear_positive"
    SMEAR_NEGATIVE = "smear_negative"
    EXTRAPULMONARY = "extrapulmonary"
