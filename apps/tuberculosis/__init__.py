from autumn.tool_kit.model_register import App

from .regions.philippines import philippines_region

app = App("tuberculosis")
app.register(philippines_region)
