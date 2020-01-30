"""
Constants used in building the AuTuMN / SUMMER models.
"""

class Compartment:
    """
    A model compartment.
    """
    SUSCEPTIBLE = "susceptible"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class Flow:
    """
    A type of flow between model compartments
    """
    INFECTION_FREQUENCY = "infection_frequency"
    INFECTION_DENSITY = "infection_density"
