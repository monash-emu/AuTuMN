from autumn import inputs
from autumn.curve import scale_up_function


def set_model_time_variant_birth_rate(model, iso3):
    birth_rates, years = inputs.get_crude_birth_rate(iso3)
    birth_rates = [b / 1000. for b in birth_rates]  # birth rates are provided / 1000 population
    model.time_variants["crude_birth_rate"] = scale_up_function(
        years, birth_rates, smoothness=0.2, method=5
    )
