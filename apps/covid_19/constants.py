from datetime import date, datetime

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


class Compartment:
    """
    A COVID-19 model compartment.
    """

    SUSCEPTIBLE = "susceptible"
    EARLY_EXPOSED = "early_exposed"
    LATE_EXPOSED = "late_exposed"
    EARLY_ACTIVE = "early_active"
    LATE_ACTIVE = "late_active"
    RECOVERED = "recovered"


"""
Compartments
"""
# People who are infectious.
INFECTIOUS_COMPARTMENTS = [
    Compartment.LATE_EXPOSED,
    Compartment.EARLY_ACTIVE,
    Compartment.LATE_ACTIVE,
]
# People who are infected, but may or may not be infectuous.
DISEASE_COMPARTMENTS = [Compartment.EARLY_EXPOSED, *INFECTIOUS_COMPARTMENTS]
# People who are eligible to receive vaccinaiton
VACCINE_ELIGIBLE_COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED]

# All model compartments
COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED, *DISEASE_COMPARTMENTS]


class Clinical:

    NON_SYMPT = "non_sympt"
    SYMPT_NON_HOSPITAL = "sympt_non_hospital"
    SYMPT_ISOLATE = "sympt_isolate"
    HOSPITAL_NON_ICU = "hospital_non_icu"
    ICU = "icu"


NOTIFICATION_STRATA = [
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]
