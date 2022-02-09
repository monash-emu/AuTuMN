class Region:
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    MANILA = "manila"
    NCR = "national-capital-region"
    CALABARZON = "calabarzon"
    CENTRAL_VISAYAS = "central-visayas"
    UNITED_KINGDOM = "united-kingdom"
    BELGIUM = "belgium"
    ITALY = "italy"
    SWEDEN = "sweden"
    FRANCE = "france"
    SPAIN = "spain"
    NORTH_EAST_METRO = "north-east-metro"
    SOUTH_EAST_METRO = "south-east-metro"
    WEST_METRO = "west-metro"
    BARWON_SOUTH_WEST = "barwon-south-west"
    GIPPSLAND = "gippsland"
    HUME = "hume"
    LODDON_MALLEE = "loddon-mallee"
    GRAMPIANS = "grampians"
    VICTORIA = "victoria"
    MARSHALL_ISLANDS = "marshall-islands"
    SABAH = "sabah"
    SELANGOR = "selangor"
    DAVAO_CITY = "davao-city"
    DAVAO_REGION = "davao-region"
    JOHOR = "johor"
    PENANG = "penang"
    KUALA_LUMPUR = "kuala-lumpur"
    NEPAL = "nepal"
    SRI_LANKA = "sri_lanka"
    BALI = "bali"
    INDONESIA = "indonesia"
    VIETNAM = "vietnam"
    HO_CHI_MINH_CITY = "ho_chi_minh_city"
    HANOI = "hanoi"
    MYANMAR = "myanmar"
    BANGLADESH = "bangladesh"
    DHAKA = "dhaka"
    COXS_BAZAR = "coxs_bazar"

    REGIONS = [
        PHILIPPINES,
        MANILA,
        NCR,
        CALABARZON,
        CENTRAL_VISAYAS,
        DAVAO_CITY,
        DAVAO_REGION,
        UNITED_KINGDOM,
        BELGIUM,
        ITALY,
        SWEDEN,
        FRANCE,
        SPAIN,
        NORTH_EAST_METRO,
        SOUTH_EAST_METRO,
        WEST_METRO,
        BARWON_SOUTH_WEST,
        GIPPSLAND,
        HUME,
        LODDON_MALLEE,
        GRAMPIANS,
        VICTORIA,
        MARSHALL_ISLANDS,
        MALAYSIA,
        SABAH,
        SELANGOR,
        JOHOR,
        PENANG,
        KUALA_LUMPUR,
        NEPAL,
        SRI_LANKA,
        INDONESIA,
        BALI,
        VIETNAM,
        HO_CHI_MINH_CITY,
        HANOI,
        MYANMAR,
        BANGLADESH,
        DHAKA,
        COXS_BAZAR,
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
        # PHILIPPINES,
        MANILA,
        # CALABARZON,
        # DAVAO_REGION,
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
        SOUTH_EAST_METRO,
        NORTH_EAST_METRO,
        WEST_METRO,
    ]

    VICTORIA_SUBREGIONS = VICTORIA_RURAL + VICTORIA_METRO

    @staticmethod
    def to_filename(name: str):
        return name.replace("-", "_")

    @staticmethod
    def to_name(filename: str):
        return filename.replace("_", "-").lower()
