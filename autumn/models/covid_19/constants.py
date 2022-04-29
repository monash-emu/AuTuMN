from autumn.settings.constants import COVID_BASE_DATETIME


class Compartment:
    """
    A COVID-19 model compartment.

    """

    SUSCEPTIBLE = "susceptible"
    EARLY_EXPOSED = "early_exposed"
    LATE_EXPOSED = "late_exposed"
    EARLY_ACTIVE = "early_active"
    LATE_ACTIVE = "late_active"


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

# All model compartments
COMPARTMENTS = [Compartment.SUSCEPTIBLE, *DISEASE_COMPARTMENTS]


"""
Stratifications.
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
    PART_WANED = "part_waned"
    WANED = "fully_waned"
    BOOSTED = "boosted"


class Strain:
    WILD_TYPE = "wild"


class Tracing:
    TRACED = "traced"
    UNTRACED = "untraced"


class History:
    NAIVE = "naive"
    EXPERIENCED = "experienced"
    WANED = "waned"


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

HOSTPIALISED_CLINICAL_STRATA = [
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
    Vaccination.PART_WANED,
    Vaccination.WANED,
    Vaccination.BOOSTED,
]

HISTORY_STRATA = [
    History.NAIVE,
    History.EXPERIENCED,
    History.WANED,
]

"""
Transitions.
"""

INFECTION = "infection"
INFECTIOUSNESS_ONSET = "infect_onset"
INCIDENCE = "incidence"
PROGRESS = "progress"
RECOVERY = "recovery"
INFECT_DEATH = "infect_death"

AGE_CLINICAL_TRANSITIONS = [INFECTIOUSNESS_ONSET, INFECT_DEATH, RECOVERY]


"""
Outputs.
"""


INFECTION_DEATHS = "infection_deaths"
NOTIFICATIONS = "notifications"  # Not a transition in the same sense as the others


"""
Mobility-related.
"""


LOCATIONS = ["home", "other_locations", "school", "work"]

GOOGLE_MOBILITY_LOCATIONS = [
    "retail_and_recreation",
    "parks",
    "workplaces",
    "transit_stations",
    "grocery_and_pharmacy",
    "residential",
    "tiles_visited",
    "single_tile",
]