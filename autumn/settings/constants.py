from datetime import datetime

COVID_BASE_DATETIME = datetime(2019, 12, 31)

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

# Age groups match the standard mixing matrices
COVID_BASE_AGEGROUPS = [str(breakpoint) for breakpoint in list(range(0, 80, 5))]
