from autumn.tool_kit.model_register import App

from .regions.victoria import victoria_region
from .regions.philippines import philippines_region
from .regions.manila import manila_region
from .regions.calabarzon import calabarzon_region
from .regions.central_visayas import central_visayas_region
from .regions.malaysia import malaysia_region
from .regions.united_kingdom import united_kingdom_region
from .regions.sweden import sweden_region
from .regions.spain import spain_region
from .regions.italy import italy_region
from .regions.france import france_region
from .regions.belgium import belgium_region
from .regions.dhhs import dhhs_region

# Used by each region to register its model.
app = App("covid_19")

# Australia
app.register(victoria_region)
app.register(dhhs_region)
# Malaysia
app.register(malaysia_region)

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
