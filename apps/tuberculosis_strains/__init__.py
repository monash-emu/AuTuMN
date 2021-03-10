from autumn.utils.model_register import App

from .regions.philippines import philippines_region

app = App("tuberculosis_strains")
app.register(philippines_region)
