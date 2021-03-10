from autumn.tool_kit.model_register import App

from .regions.philippines import philippines_region
from .regions.victoria import victoria_region

app = App("sir_example")
app.register(philippines_region)
app.register(victoria_region)
