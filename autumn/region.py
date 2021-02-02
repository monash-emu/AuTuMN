class Region:
    AUSTRALIA = "australia"
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    VICTORIA = "victoria"
    NSW = "nsw"
    LIBERIA = "liberia"
    MANILA = "manila"
    CALABARZON = "calabarzon"
    BICOL = "bicol"
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

    REGIONS = [
        AUSTRALIA,
        PHILIPPINES,
        MALAYSIA,
        VICTORIA,
        NSW,
        LIBERIA,
        MANILA,
        CALABARZON,
        BICOL,
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
        CENTRAL_VISAYAS,
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
