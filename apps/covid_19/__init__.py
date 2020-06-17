from autumn.constants import Region
from autumn.tool_kit.model_register import App

from .app import RegionApp

app = App()
app.register(RegionApp(Region.AUSTRALIA))
app.register(RegionApp(Region.PHILIPPINES))
app.register(RegionApp(Region.MALAYSIA))
app.register(RegionApp(Region.VICTORIA))
app.register(RegionApp(Region.LIBERIA))
app.register(RegionApp(Region.MANILA))
app.register(RegionApp(Region.CALABARZON))
app.register(RegionApp(Region.BICOL))
app.register(RegionApp(Region.CENTRAL_VISAYAS))
# app.register(RegionApp(Region.UNITED_KINGDOM))

# Functions and data exposed to the outside world
REGION_APPS = app.region_names
get_region_app = app.get_region_app
