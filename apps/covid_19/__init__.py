from autumn.tool_kit.model_register import App

from .regions.philippines import philippines_region
from .regions.manila import manila_region
from .regions.calabarzon import calabarzon_region
from .regions.central_visayas import central_visayas_region
from .regions.malaysia import malaysia_region
from .regions.sabah import sabah_region
from .regions.united_kingdom import united_kingdom_region
from .regions.sweden import sweden_region
from .regions.spain import spain_region
from .regions.italy import italy_region
from .regions.france import france_region
from .regions.belgium import belgium_region
from .regions.north_metro import north_metro_region
from .regions.south_east_metro import south_east_metro_region
from .regions.south_metro import south_metro_region
from .regions.west_metro import west_metro_region
from .regions.barwon_south_west import barwon_south_west_region
from .regions.gippsland import gippsland_region
from .regions.hume import hume_region
from .regions.loddon_mallee import loddon_mallee_region
from .regions.grampians import grampians_region


# Used by each region to register its model.
app = App("covid_19")

# Australia
app.register(north_metro_region)
app.register(south_east_metro_region)
app.register(south_metro_region)
app.register(west_metro_region)
app.register(barwon_south_west_region)
app.register(gippsland_region)
app.register(hume_region)
app.register(loddon_mallee_region)
app.register(grampians_region)


# Malaysia
app.register(malaysia_region)
app.register(sabah_region)

# Philippines regions
app.register(philippines_region)
app.register(manila_region)
app.register(calabarzon_region)
app.register(central_visayas_region)

# Mixing optimization regions
app.register(belgium_region)
app.register(united_kingdom_region)
app.register(italy_region)
app.register(france_region)
app.register(sweden_region)
app.register(spain_region)
