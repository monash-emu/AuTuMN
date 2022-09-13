class Region:
    wpro_australia = "australia"
    wpro_malaysia = "malaysia"
    wpro_philippines = "philippines"
    wpro_CHINA = "china"
    wpro_MONGOLIA = "mongolia"
    wpro_NEW_ZEALAND = "new-zealand"
    wpro_SOUTH_KOREA = "south-korea"
    wpro_VIET_NAM = "viet-nam"
    wpro_SINGAPORE = "singapore"
    wpro_JAPAN = "japan"
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
    MARSHALL_ISLANDS = "marshall-islands"
    SABAH = "sabah"
    SELANGOR = "selangor"
    DAVAO_CITY = "davao-city"
    DAVAO_REGION = "davao-region"
    JOHOR = "johor"
    PENANG = "penang"
    KUALA_LUMPUR = "kuala-lumpur"
    SRI_LANKA = "sri_lanka"
    BALI = "bali"
    VIETNAM = "vietnam"
    HO_CHI_MINH_CITY = "ho_chi_minh_city"
    HANOI = "hanoi"
    MYANMAR = "myanmar"
    BANGLADESH = "bangladesh"
    DHAKA = "dhaka"
    COXS_BAZAR = "coxs_bazar"
    BHUTAN = "bhutan"
    THIMPHU = "thimphu"
    MULTI = "multi"
    KIRIBATI = "kiribati"
    NORTHERN_TERRITORY = "northern-territory"
    AUSTRALIA = "australia"

    REGIONS = [
        wpro_australia,
        wpro_philippines,
        wpro_malaysia,
        wpro_CHINA,
        wpro_MONGOLIA,
        wpro_NEW_ZEALAND,
        wpro_SOUTH_KOREA,
        wpro_VIET_NAM,
        wpro_SINGAPORE,
        wpro_JAPAN,
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
        MARSHALL_ISLANDS,
        MALAYSIA,
        SABAH,
        SELANGOR,
        JOHOR,
        PENANG,
        KUALA_LUMPUR,
        SRI_LANKA,
        BALI,
        VIETNAM,
        HO_CHI_MINH_CITY,
        HANOI,
        MYANMAR,
        BANGLADESH,
        DHAKA,
        COXS_BAZAR,
        BHUTAN,
        THIMPHU,
        MULTI,
        KIRIBATI,
        NORTHERN_TERRITORY,
        AUSTRALIA,
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

    WPRO_REGIONS = [
        wpro_australia,
        wpro_philippines,
        wpro_malaysia
    ]

    SCHOOL_PROJECT_REGIONS = [PHILIPPINES, FRANCE, AUSTRALIA]
    @staticmethod
    def to_filename(name: str):
        return name.replace("-", "_")

    @staticmethod
    def to_name(filename: str):
        return filename.replace("_", "-").lower()
