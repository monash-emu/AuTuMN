"""
Constants used in building the AuTuMN / SUMMER models.
"""
import os

# Filesystem paths
file_path = os.path.abspath(__file__)
separator = "\\" if "\\" in file_path else "/"
BASE_PATH = separator.join(file_path.split(separator)[:-2])
DATA_PATH = os.path.join(BASE_PATH, "data")
EXCEL_PATH = os.path.join(DATA_PATH, "xls")


class Compartment:
    """
    A model compartment.
    """

    SUSCEPTIBLE = "susceptible"
    EARLY_LATENT = "early_latent"
    LATE_LATENT = "late_latent"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"


class Flow:
    """
    A type of flow between model compartments
    """

    CUSTOM = "customised_flows"
    STANDARD = "standard_flows"
    INFECTION_FREQUENCY = "infection_frequency"
    INFECTION_DENSITY = "infection_density"
    COMPARTMENT_DEATH = "compartment_death"


class BirthApproach:
    """
    Options for birth rate settings in model
    """

    NO_BIRTH = "no_birth"
    ADD_CRUDE = "add_crude_birth_rate"
    REPLACE_DEATHS = "replace_deaths"


class IntegrationType:
    """
    Options for ODE solver used by model
    """

    ODE_INT = "odeint"
    SOLVE_IVP = "solve_ivp"


class Stratification:
    """
    Attribute used to stratify the population within compartments. 
    """

    AGE = "age"
    STRAIN = "strain"
    LOCATION = "location"
