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
        self.all_stratifications = {}
        # lateXagegroup_75Xclinical_sympt_non_hospital
        for compartment_name in self.compartment_names:
            # ['late', 'agegroup_75', 'clinical_sympt_non_hospital']
            parts = compartment_name.split("X")
            # ['agegroup_75', 'clinical_sympt_non_hospital']
            strats = parts[1:]
            for strat in strats:
                # clinical_sympt_non_hospital
                parts = strat.split("_")
                strat_name = parts[0]
                strata = "_".join(parts[1:])
                if strat_name not in self.all_stratifications:
                    self.all_stratifications[strat_name] = []
                if strata not in self.all_stratifications[strat_name]:
                    self.all_stratifications[strat_name].append(strata)
