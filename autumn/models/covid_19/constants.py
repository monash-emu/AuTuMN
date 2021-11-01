from datetime import date, datetime

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


class Compartment:
    """
    A COVID-19 model compartment
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

# People who are infectious
INFECTIOUS_COMPARTMENTS = [
    Compartment.LATE_EXPOSED,
    Compartment.EARLY_ACTIVE,
    Compartment.LATE_ACTIVE,
]
# People who are infected, but may or may not be infectious
DISEASE_COMPARTMENTS = [Compartment.EARLY_EXPOSED, *INFECTIOUS_COMPARTMENTS]

# People who are eligible to receive vaccination
VACCINE_ELIGIBLE_COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED]

# All model compartments
COMPARTMENTS = [Compartment.SUSCEPTIBLE, Compartment.RECOVERED, *DISEASE_COMPARTMENTS]


"""
Stratifications
"""

# Age groups match the standard mixing matrices
AGEGROUP_STRATA = [str(breakpoint) for breakpoint in list(range(0, 80, 5))]


class Clinical:
    NON_SYMPT = "non_sympt"
    SYMPT_NON_HOSPITAL = "sympt_non_hospital"
    SYMPT_ISOLATE = "sympt_isolate"
    HOSPITAL_NON_ICU = "hospital_non_icu"
    ICU = "icu"


class Vaccination:
    UNVACCINATED = "unvaccinated"
    ONE_DOSE_ONLY = "one_dose"
    VACCINATED = "fully_vaccinated"


class Strain:
    WILD_TYPE = "wild"


class Tracing:
    TRACED = "traced"
    UNTRACED = "untraced"


class History:
    NAIVE = "naive"
    EXPERIENCED = "experienced"


CLINICAL_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.SYMPT_NON_HOSPITAL,
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

NOTIFICATION_CLINICAL_STRATA = [
    Clinical.SYMPT_ISOLATE,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

FIXED_STRATA = [
    Clinical.NON_SYMPT,
    Clinical.HOSPITAL_NON_ICU,
    Clinical.ICU,
]

VACCINATION_STRATA = [
    Vaccination.UNVACCINATED,
    Vaccination.ONE_DOSE_ONLY,
    Vaccination.VACCINATED,
]

HISTORY_STRATA = [
    History.NAIVE,
    History.EXPERIENCED,
]

"""
Transitions
"""

INFECTION = "infection"
INFECTIOUSNESS_ONSET = "infect_onset"
INCIDENCE = "incidence"
NOTIFICATIONS = "notifications"  # Not a transition in the same sense as the others
PROGRESS = "progress"
RECOVERY = "recovery"
INFECT_DEATH = "infect_death"

AGE_CLINICAL_TRANSITIONS = [INFECTIOUSNESS_ONSET, INFECT_DEATH, RECOVERY]


"""
Vic model options
"""


class VicModelTypes:
    NON_VIC = "non_vic"
    VIC_SUPER_2020 = "vic_super_2020"
    VIC_SUPER_2021 = "vic_super_2021"
    VIC_REGION_2021 = "vic_region_2021"


VIC_MODEL_OPTIONS = [
    VicModelTypes.NON_VIC,
    VicModelTypes.VIC_SUPER_2020,
    VicModelTypes.VIC_SUPER_2021,
    VicModelTypes.VIC_REGION_2021,
]

GOOGLE_MOBILITY_LOCATIONS = [
    "retail_and_recreation",
    "parks",
    "workplaces",
    "transit_stations",
    "grocery_and_pharmacy",
    "residential"
]
