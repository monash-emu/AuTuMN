from .countries import Country, CountryModel

aus = CountryModel(Country.AUSTRALIA)
phl = CountryModel(Country.PHILIPPINES)

COUNTRY_RUNNERS = [
    "aus",
    "phl",
]
