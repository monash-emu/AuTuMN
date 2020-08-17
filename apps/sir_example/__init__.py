from autumn.tool_kit.model_register import App

from .regions.australia import australia_region
from .regions.philippines import philippines_region

app = App("sir_example")
app.register(philippines_region)
app.register(australia_region)
