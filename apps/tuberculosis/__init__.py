from autumn.utils.model_register import App

from .regions.philippines import philippines_region
from .regions.marshall_islands import marshall_islands_region

app = App("tuberculosis")
app.register(philippines_region)
app.register(marshall_islands_region)
