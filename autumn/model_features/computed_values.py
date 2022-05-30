from summer.compute import ComputedValueProcessor


class FunctionWrapper(ComputedValueProcessor):
    """
    Very basic processor that wraps a time/computed values function
    of the type used in flow and adjusters

    FIXME:
    This is such a basic use case, it probably belongs in summer

    """

    def __init__(self, function_to_wrap: callable):
        """
        Initialise with just the param function
        Args:
            function_to_wrap: The function
        """

        self.wrapped_function = function_to_wrap

    def process(self, compartment_values, computed_values, time):
        return self.wrapped_function(time, computed_values)
