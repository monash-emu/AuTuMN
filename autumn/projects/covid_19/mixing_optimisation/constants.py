from autumn.settings import Region

CALIBRATION_START = 100
CALIBRATION_END = 244


# Please keep this code in place
OPTI_REGIONS = [
    Region.BELGIUM,
    Region.FRANCE,
    Region.ITALY,
    Region.SPAIN,
    Region.SWEDEN,
    Region.UNITED_KINGDOM,
]

OPTI_ISO3S = [
    "BEL",
    "GBR",
    "ITA",
    "SWE",
    "FRA",
    "ESP",
]

COUNTRY_TITLES = {
    Region.BELGIUM: "Belgium",
    Region.FRANCE: "France",
    Region.ITALY: "Italy",
    Region.SPAIN: "Spain",
    Region.SWEDEN: "Sweden",
    Region.UNITED_KINGDOM: "United Kingdom",
}


#  Definitions of the three phases
PHASE_2_START_TIME = 275  # start on 1 October
DURATION_PHASES_2_AND_3 = 365 + 90

PHASE_2_DURATION = {
    "six_months": 183,
    "twelve_months": 365,
}

# Mixing factor bounds
MIXING_FACTOR_BOUNDS = [0.1, 1.0]

# Microdistancing
MICRODISTANCING_OPTI_PARAMS = {
    # apply mild microdistancing
    "behaviour": {
        "parameters": {
            "start_asymptote": 0.10,
            "end_asymptote": 0.10,
        }
    },
    # remove the behaviour adjuster's effect
    "behaviour_adjuster": {"parameters": {"start_asymptote": 1.0, "end_asymptote": 1.0}},
}
