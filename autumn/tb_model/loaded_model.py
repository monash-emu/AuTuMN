import numpy


class LoadedModel:
    """
    A model placeholder, used to store the outputs of a previous model run.
    """

    def __init__(self, outputs, derived_outputs):
        self.compartment_names = [
            name for name in outputs.keys() if name not in ["idx", "Scenario", "times"]
        ]

        self.outputs = numpy.column_stack(
            [
                list(column.values())
                for name, column in outputs.items()
                if name not in ["idx", "Scenario", "times"]
            ]
        )
        self.derived_outputs = (
            {
                key: list(value.values())
                for key, value in derived_outputs.items()
                if key not in ["idx", "Scenario", "times"]
            }
            if derived_outputs is not None
            else None
        )

        self.times = list(outputs["times"].values())
