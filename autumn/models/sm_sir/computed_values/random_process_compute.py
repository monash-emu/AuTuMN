from summer.compute import ComputedValueProcessor


class RandomProcessProc(ComputedValueProcessor):
    """
    Calculate the values of the random process
    """

    def __init__(self, rp_time_variant_func):
        self.rp_time_variant_func = rp_time_variant_func

    def process(self, compartment_values, computed_values, time):
        return self.rp_time_variant_func(time)
