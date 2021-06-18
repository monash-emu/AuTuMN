class Region:
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    VICTORIA = "victoria"
    MANILA = "manila"
    CALABARZON = "calabarzon"
    CENTRAL_VISAYAS = "central-visayas"
    UNITED_KINGDOM = "united-kingdom"
    BELGIUM = "belgium"
    ITALY = "italy"
    SWEDEN = "sweden"
    FRANCE = "france"
    SPAIN = "spain"
    NORTH_METRO = "north-metro"
    SOUTH_EAST_METRO = "south-east-metro"
    SOUTH_METRO = "south-metro"
    WEST_METRO = "west-metro"
    BARWON_SOUTH_WEST = "barwon-south-west"
    GIPPSLAND = "gippsland"
    HUME = "hume"
    LODDON_MALLEE = "loddon-mallee"
    GRAMPIANS = "grampians"
    MARSHALL_ISLANDS = "marshall-islands"
    SABAH = "sabah"
    SELANGOR = "selangor"
    DAVAO_CITY = "davao-city"
    JOHOR = "johor"
    PENANG = "penang"
    KUALA_LUMPUR = "kuala-lumpur"
    NEPAL = "nepal"
    SRI_LANKA = "sri_lanka"
    BALI = "bali"

    REGIONS = [
        PHILIPPINES,
        MALAYSIA,
        VICTORIA,
        MANILA,
        CALABARZON,
        CENTRAL_VISAYAS,
        UNITED_KINGDOM,
        BELGIUM,
        ITALY,
        SWEDEN,
        FRANCE,
        SPAIN,
        NORTH_METRO,
        SOUTH_EAST_METRO,
        SOUTH_METRO,
        WEST_METRO,
        BARWON_SOUTH_WEST,
        GIPPSLAND,
        HUME,
        LODDON_MALLEE,
        GRAMPIANS,
        MARSHALL_ISLANDS,
        SABAH,
        SELANGOR,
        JOHOR,
        PENANG,
        KUALA_LUMPUR,
        DAVAO_CITY,
        NEPAL,
        SRI_LANKA,
        BALI,
    ]

    MALAYSIA_REGIONS = [
        MALAYSIA,
        SELANGOR,
        JOHOR,
        PENANG,
        KUALA_LUMPUR,
    ]

    MIXING_OPTI_REGIONS = [
        UNITED_KINGDOM,
        BELGIUM,
        ITALY,
        SWEDEN,
        FRANCE,
        SPAIN,
    ]

    PHILIPPINES_REGIONS = [
        PHILIPPINES,
        MANILA,
        CALABARZON,
        # CENTRAL_VISAYAS,
        # DAVAO_CITY,
    ]

    VICTORIA_RURAL = [
        BARWON_SOUTH_WEST,
        GIPPSLAND,
        HUME,
        LODDON_MALLEE,
        GRAMPIANS,
    ]

    VICTORIA_METRO = [
        NORTH_METRO,
        SOUTH_EAST_METRO,
        SOUTH_METRO,
        WEST_METRO,
    ]

    VICTORIA_SUBREGIONS = VICTORIA_RURAL + VICTORIA_METRO

    @staticmethod
    def to_filename(name: str):
        return name.replace("-", "_")

    @staticmethod
    def to_name(filename: str):
        return filename.replace("_", "-").lower()
