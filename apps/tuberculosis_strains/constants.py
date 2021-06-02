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
