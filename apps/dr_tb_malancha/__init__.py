from autumn.constants import Region
from autumn.tool_kit.model_register import App

from .app import RegionApp

app = App()
app.register(RegionApp(Region.AUSTRALIA))
app.register(RegionApp(Region.PHILIPPINES))

# Functions and data exposed to the outside world
REGION_APPS = app.region_names
get_region_app = app.get_region_app
