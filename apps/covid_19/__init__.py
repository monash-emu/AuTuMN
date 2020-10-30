from autumn.tool_kit.model_register import App

from .regions.united_kingdom import united_kingdom_region
from .regions.sweden import sweden_region
from .regions.spain import spain_region
from .regions.italy import italy_region
from .regions.france import france_region
from .regions.belgium import belgium_region



# Used by each region to register its model.
app = App("covid_19")

# Mixing optimization regions
app.register(belgium_region)
app.register(united_kingdom_region)
app.register(italy_region)
app.register(france_region)
app.register(sweden_region)
app.register(spain_region)
