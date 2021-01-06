"""
This module presents classes which are used to define stratifications, which can be applied to the model.
"""
from typing import List, Dict, Callable, Union, Optional

import numpy as np

from summer2.compartment import Compartment
from summer2.adjust import Multiply, Overwrite


Adjustment = Union[Multiply, Overwrite]
MixingMatrix = Union[np.ndarray, Callable[[float], np.ndarray]]


class Stratification:
    """
    A generic stratification applied to the compartmental model.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.

    """

    _is_ageing = False
    _is_strain = False

    def __init__(
        self,
        name: str,
        strata: List[str],
        compartments: List[str],
    ):
        self.name = name
        self.strata = list(map(str, strata))
        self.compartments = [Compartment(c) if type(c) is str else c for c in compartments]

        # Split population evently by default.
        num_strata = len(self.strata)
        self.population_split = {s: 1 / num_strata for s in self.strata}

        # Flows are not adjusted by default.
        self.flow_adjustments = {}
        self.infectiousness_adjustments = {}

        # No heterogeneous mixing matrix by default.
        self.mixing_matrix = None

    def is_ageing(self) -> bool:
        """Returns ``True`` if this stratification represets a set of age groups with ageing dynamics"""
        return self._is_ageing

    def is_strain(self) -> bool:
        """Returns ``True`` if this stratification represets a set of disease strains"""
        return self._is_strain

    def set_population_split(self, proportions: Dict[str, float]):
        """
        Sets how the stratification will split the population between the strata.

        Args:
            proportions: A map of proportions to be assigned to each strata.

        The supplied proportions must:

            - Specify all strata
            - All be positive
            - Sum to 1 +/- 0.01

        """
        msg = f"All strata must be specified when setting population split: {proportions}"
        assert list(proportions.keys()) == self.strata, msg
        msg = f"All proportions must be >= 0 when setting population split: {proportions}"
        assert all([v >= 0 for v in proportions.values()]), msg
        msg = f"All proportions sum to 1+/-0.01 when setting population split: {proportions}"
        assert abs(1 - sum(proportions.values())) < 1e2, msg
        self.population_split = proportions

    def add_flow_adjustments(
        self,
        flow_name: str,
        adjustments: Dict[str, Adjustment],
        source_strata: Optional[Dict[str, str]] = None,
        dest_strata: Optional[Dict[str, str]] = None,
    ):
        """
        Add an adjustment of a flow to the stratification.
        You can use time-varying functions for infectiousness adjustments.

        It is possible to specify multiple conflicting flow adjustments for the same flow.
        In this case, only the last-created applicable adjustment will be chosen.

        Args:
            flow_name: The name of the flow to adjust.
            adjustments: An dict of adjustments to apply to the flow.
            source_strata (optional): A whitelist of strata to filter the target flow's source compartments.
            dest_strata (optional): A whitelist of strata to filter the target flow's destination compartments.

        Example:
            Create an adjustment for the 'recovery' flow based on location::

                strat = Stratification(
                    name="location",
                    strata=["urban", "rural", "alpine"],
                    compartments=["S", "I", "R"]
                )
                strat.add_flow_adjustments("recovery", {
                    "urban": Multiply(1.5),
                    "rural": Multiply(0.8),
                    "alpine": None, # No adjustment
                })


        """
        source_strata = source_strata or {}
        dest_strata = dest_strata or {}
        msg = "You must specify all strata when adding flow adjustments."
        assert set(adjustments.keys()) == set(self.strata), msg

        assert all(
            [type(a) is Overwrite or type(a) is Multiply or a is None for a in adjustments.values()]
        ), "All flow adjustments must be Multiply, Overwrite or None."

        if flow_name not in self.flow_adjustments:
            self.flow_adjustments[flow_name] = []

        self.flow_adjustments[flow_name].append((adjustments, source_strata, dest_strata))

    def get_flow_adjustment(self, flow) -> dict:
        """
        Returns the most recently added flow adjustment that matches a given flow.
        """
        flow_adjustments = self.flow_adjustments.get(flow.name, [])
        matching_adjustment = None
        for adjustment, source_strata, dest_strata in flow_adjustments:
            # Skip any adjustments that don't match the flow's source compartment strata.
            msg = f"Source strata specified in flow adjustment of {self.name} but {flow.name} does not have a source"
            assert not (source_strata and not flow.source), msg
            if source_strata and flow.source and not flow.source.has_strata(source_strata):
                continue

            # Skip any adjustments that don't match the flow's dest compartment strata.
            msg = f"Dest strata specified in flow adjustment of {self.name} but {flow.name} does not have a dest"
            assert not (dest_strata and not flow.dest), msg
            if dest_strata and flow.dest and not flow.dest.has_strata(dest_strata):
                continue

            matching_adjustment = adjustment

        return matching_adjustment

    def add_infectiousness_adjustments(
        self, compartment_name: str, adjustments: Dict[str, Adjustment]
    ):
        """
        Add an adjustment of a compartment's infectiousness to the stratification.
        You cannot use time-varying functions for infectiousness adjustments.

        Args:
            compartment_name: The name of the compartment to adjust.
            adjustments: An dict of adjustments to apply to the compartment.

        Example:
            Create an adjustment for the 'I' compartment based on location::

                strat = Stratification(
                    name="location",
                    strata=["urban", "rural", "alpine"],
                    compartments=["S", "I", "R"]
                )
                strat.add_infectiousness_adjustments("I", {
                    "urban": adjust.Multiply(1.5),
                    "rural": adjust.Multiply(0.8),
                    "alpine": None, # No adjustment
                })

        """
        msg = "You must specify all strata when adding infectiousness adjustments."
        assert set(adjustments.keys()) == set(self.strata), msg

        assert all(
            [type(a) is Overwrite or type(a) is Multiply or a is None for a in adjustments.values()]
        ), "All infectiousness adjustments must be Multiply, Overwrite or None."

        msg = f"An infectiousness adjustment for {compartment_name} already exists for strat {self.name}"
        assert compartment_name not in self.infectiousness_adjustments, msg

        msg = "Cannot use time varying functions for infectiousness adjustments."
        assert not any([callable(a.param) for a in adjustments.values() if a])

        self.infectiousness_adjustments[compartment_name] = adjustments

    def set_mixing_matrix(self, mixing_matrix: MixingMatrix):
        """"""
        msg = "Strain stratifications cannot have a mixing matrix."
        assert not self.is_strain(), msg

        mm = mixing_matrix(0) if callable(mixing_matrix) else mixing_matrix

        msg = "Mixing matrix must be a NumPy array, or return a NumPy array."
        assert type(mm) is np.ndarray, msg

        num_strata = len(self.strata)
        msg = f"Mixing matrix must have shape ({num_strata}, {num_strata})"
        assert mm.shape == (num_strata, num_strata), msg

        self.mixing_matrix = mixing_matrix

    def _stratify_compartments(self, comps: List[Compartment]) -> List[Compartment]:
        """
        Stratify the model compartments into sub-compartments, based on the strata names,
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartments.
        """
        new_comps = []
        for old_comp in comps:
            should_stratify = old_comp.has_name_in_list(self.compartments)
            if should_stratify:
                for stratum in self.strata:
                    new_comp = old_comp.stratify(self.name, stratum)
                    new_comps.append(new_comp)
            else:
                new_comps.append(old_comp)

        return new_comps

    def _stratify_compartment_values(
        self, comps: List[Compartment], comp_values: np.ndarray
    ) -> np.ndarray:
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided,
        Split the population according to the provided proportions.
        Only compartments specified in the stratification's definition will be stratified.
        Returns the new compartment values.
        """
        assert len(comps) == len(comp_values)
        new_comp_values = []
        for idx in range(len(comp_values)):
            should_stratify = comps[idx].has_name_in_list(self.compartments)
            if should_stratify:
                for stratum in self.strata:
                    new_value = comp_values[idx] * self.population_split[stratum]
                    new_comp_values.append(new_value)
            else:
                new_comp_values.append(comp_values[idx])

        return np.array(new_comp_values)


class AgeStratification(Stratification):
    """
    A stratification that represets a set of age groups with ageing dynamics.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.


    Strata must be a list of strings or integers that represent all ages. For example, the strata
    [0, 10, 20, 30] will be interpreted as the age groups 0-9, 10-19, 20-29, 30+ respectively.

    Using this stratification will automatically add a set of ageing flows to the model,
    where the exit rate is proportional to the age group's time span. For example, in the 0-9 strata,
    there will be a flow created where 10% of occupants "age up" into the 10-19 strata each year.
    **Critically**, this feature assumes that each of the model's time steps represents a year.
    """

    _is_ageing = True

    def __init__(
        self,
        name: str,
        strata: List[str],
        compartments: List[str],
    ):
        try:
            _strata = sorted(map(int, strata))
        except:
            raise AssertionError("Strata much be a int-compatible")

        assert _strata[0] == 0, "First age strata must be 0"

        _strata = map(str, _strata)
        super().__init__(name, _strata, compartments)


class StrainStratification(Stratification):
    """
    A stratification that represets a set of disease strains.

    Args:
        name: The name of the stratification.
        strata: The new strata that will be created.
        compartments: The compartments which will be stratified.


    Each supplied strata will be interpreted as a different strain of the disease being modelled.
    This will mean that the force of infection calculations will consider each strain separately.

    Strain stratifications cannot use a mixing matrix
    """

    _is_strain = True
