from abc import ABC, abstractmethod
from typing import Callable


class Adjustment:
    """
    NOTE: THIS IS NOT USED ANYWHERE, JUST A CONCEPT FOR NOW.

    An stratification-based adjustment to a value in the model.
    Can be applied to values such as flow parameters, infectiousness levels.
    For example, we might want to vary the infection fataility rate by age group.
    This would be done when applying a age-based stratification.
    """

    def __init__(self, adjustments: list, include=[], exclude=[]):
        """
        Create a new adjustment.

        The adjustments are applied to all strata by default.
        The user can specify which strata to apply the adjustments to using `include` and `exclude`.
        The user can specify include, exclude or neither, but not both include and exclude.

        include: a list of existing strata to explicitly include when applying adjustments.
        exclude: a list of exiting strata to explicitly exclude when applying adjustments.
        adjustments: the adjustments to apply to the new strata
         
        Example: Set varying values based on age for people at home or work
                
            Adjustment(
                include = [{"location": "home"}, {"location": "work"}],
                adjustments = [
                    Multiply('age', '0', 0.1),
                    Multiply('age', '5', 0.2),
                    Multiply('age', '10', 0.3),
                    Multiply('age', '15', 0.4),
                ]
            )

        Include and exclude identify relevant strata by:
            - Using intersection logic for each item in the list
            - Using union logic between each list element

        For example: apply to all strata that are (home AND age 50) OR (work)

            include = [{"location": "home", "age": "50"}, {"location": "work"}]

        """
        assert adjustments, "No adjustments specified."
        assert not (include and exclude), "Cannot use `include` and `exclude` in Adjustment."
        self.adjustments = adjustments
        self.include = include
        self.exclude = exclude

    @staticmethod
    def multiply_all(strata_name: str, adjustment_data: list, include=[], exclude=[]):
        adjs = [Multiply(strata_name, stratum, value) for stratum, value in adjustment_data.items()]
        return Adjustment(adjs, include, exclude)


class BaseAdjustment(ABC):
    @abstractmethod
    def apply(self, prior_value: float, time: float) -> float:
        pass


class Overwrite(BaseAdjustment):
    """
    Adjust value by overwriting prior values.
    """

    def __init__(self, strat_name: str, stratum: str, value: float):
        self.strat_name = strat_name
        self.stratum = stratum
        self.value = value

    def apply(self, prior_value: float, time: float):
        return self.value


class Compose(BaseAdjustment):
    """
    Adjust value by multiplying with a function of time.
    """

    def __init__(self, strat_name: str, stratum: str, func: Callable[[float], float]):
        self.strat_name = strat_name
        self.stratum = stratum
        self.func = func

    def apply(self, prior_value: float, time: float):
        return prior_value * self.func(time)


class Multiply(BaseAdjustment):
    """
    Adjust value by multiplying with a constant.
    """

    def __init__(self, strat_name: str, stratum: str, value: float):
        self.strat_name = strat_name
        self.stratum = stratum
        self.value = value

    def apply(self, prior_value: float, time: float):
        return prior_value * self.value
