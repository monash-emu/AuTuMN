from .countries import Country, CountryModel

aus = CountryModel(Country.AUSTRALIA)
phl = CountryModel(Country.PHILIPPINES)
mys = CountryModel(Country.MALAYSIA)
vic = CountryModel(Country.VICTORIA)
lbr = CountryModel(Country.LIBERIA)
man = CountryModel(Country.MANILA)
cal = CountryModel(Country.CALABARZON)
COUNTRY_RUNNERS = ["aus", "phl", "mys", "vic", "lbr", "man", "cal"]
