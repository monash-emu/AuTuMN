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


class ClinicalStratum:

    NON_SYMPT = "non_sympt"
    SYMPT_NON_HOSPITAL = "sympt_non_hospital"
    SYMPT_ISOLATE = "sympt_isolate"
    HOSPITAL_NON_ICU = "hospital_non_icu"
    ICU = "icu"
