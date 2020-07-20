from typing import List, Tuple, Dict

import numpy as np

from summer.constants import FlowAdjustment
from summer.compartment import Compartment

AGE_STRATA_NAME = "age"
OVERWRITE_CHARACTER = "W"


class Stratification:
    """
    A stratification applied to a the compartmental model.
    """

    def __init__(
        self,
        name: str,
        strata: List[str],
        compartments: List[str],
        comp_split_props: Dict[str, float],
        flow_adjustments: Dict[str, Dict[str, float]],
    ):
        self.name = name
        self.strata = list(map(str, strata))
        self.comp_split_props = get_all_proportions(self.strata, comp_split_props)
        self.flow_adjustments = parse_flow_adjustments(flow_adjustments)
        self.compartments = [Compartment(c) if type(c) is str else c for c in compartments]

    def is_ageing(self) -> bool:
        return self.name == AGE_STRATA_NAME

    def get_flow_adjustment(self, comp: Compartment, stratum: str, param_name: str):
        """
        Returns an adjustment tuple or None
        """
        param_adjs = self.flow_adjustments.get(param_name)
        if not param_adjs:
            # No adjustments for this parameter.
            return

        comp_adj = None
        for _comp_adj in param_adjs:
            if _comp_adj["strata"] == comp._strat_values:
                comp_adj = _comp_adj["adjustments"]

        if not comp_adj:
            # No adjustments for this compartment.
            return

        return comp_adj.get(stratum)

    def update_mixing_categories(
        self, old_mixing_categories: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Returns a new list of mixing categories, with this stratification included.      
        """
        new_mc = []
        for mc in old_mixing_categories:
            for stratum in self.strata:
                new_mc.append({**mc, self.name: stratum})

        return new_mc


def parse_flow_adjustments(flow_adjustments: Dict[str, Dict[str, float]]):
    """
    INPUT
    {
        "to_infectiousXagegroup_50": {
            <empty>: just use the non-adjusted version with the old name 
            "non_symptW": 0.51, # Overwrite to get stratified verion
            "clinical": 0.51, # Multiply to get stratified verion
            "icu": "prop_sympt_non_hospital_50", # Apply a function to previous parameter to get stratified verion
        },
    }
    OUTPUT
    {
        "to_infectious": [
            {
                "strata": {"agegroup": "50"},
                "adjustments": {
                    "non_sympt": (FlowAdjustment.OVERWRITE, 0.51),
                    "clinical": (FlowAdjustment.MULTIPLY, 0.51),
                    "icu": (FlowAdjustment.COMPOSE, "prop_sympt_non_hospital_50"),
                }
            }
        ]
    }
    """
    parsed_flow_adjustments = {}
    for adjust_key, adjustments in flow_adjustments.items():
        parts = adjust_key.split("X")
        param_name = parts[0]
        if param_name not in parsed_flow_adjustments:
            parsed_flow_adjustments[param_name] = []

        parsed_strata = {}
        for strat in parts[1:]:
            strat_parts = strat.split("_")
            parsed_strata[strat_parts[0]] = "_".join(strat_parts[1:])

        parsed_adjustments = {}
        for k, v in adjustments.items():
            if type(v) is str:
                parsed_adjustments[k] = (FlowAdjustment.COMPOSE, v)
            elif k.endswith(OVERWRITE_CHARACTER):
                key = k[:-1]
                parsed_adjustments[key] = (FlowAdjustment.OVERWRITE, v)
            else:
                parsed_adjustments[k] = (FlowAdjustment.MULTIPLY, v)

        parsed_adjustment = {
            "strata": parsed_strata,
            "adjustments": parsed_adjustments,
        }
        parsed_flow_adjustments[param_name].append(parsed_adjustment)

    return parsed_flow_adjustments


def get_all_proportions(strata_names: List[str], strata_proportions: Dict[str, float]):
    """
    Determine what % of population get assigned to the different strata.
    """
    proportion_allocated = sum(strata_proportions.values())
    remaining_names = [n for n in strata_names if n not in strata_proportions]
    count_remaining = len(remaining_names)
    assert set(strata_proportions.keys()).issubset(strata_names), "Invalid proprotion keys"
    assert proportion_allocated <= 1, "Sum of proportions must not exceed 1.0"
    if not remaining_names:
        eps = 1e-12
        assert 1 - proportion_allocated < eps, "Sum of proportions must be 1.0"
        return strata_proportions
    else:
        # Divide the remaining proprotions equally
        starting_proportion = (1 - proportion_allocated) / count_remaining
        remaining_proportions = {name: starting_proportion for name in remaining_names}
        return {**strata_proportions, **remaining_proportions}


def get_stratified_compartment_names(
    strat: Stratification, comp_names: List[Compartment],
) -> List[Compartment]:
    """
    Stratify the model compartments into sub-compartments, based on the strata names provided,
    Only compartments specified in `comps_to_stratify` will be stratified.
    Returns the new compartment names.
    """
    new_comps = []
    for old_comp in comp_names:
        should_stratify = old_comp.has_name_in_list(strat.compartments)
        if should_stratify:
            for stratum in strat.strata:
                new_comp = old_comp.stratify(strat.name, stratum)
                new_comps.append(new_comp)
        else:
            new_comps.append(old_comp)

    return new_comps


def get_stratified_compartment_values(
    strat: Stratification, comp_names: List[Compartment], comp_values: np.ndarray,
) -> np.ndarray:
    """
    Stratify the model compartments into sub-compartments, based on the strata names provided,
    Split the population according to the provided proprotions.
    Only compartments specified in `comps_to_stratify` will be stratified.
    Returns the new compartment names and values
    """
    assert len(comp_names) == len(comp_values)
    new_comp_values = []
    for idx in range(len(comp_values)):
        should_stratify = comp_names[idx].has_name_in_list(strat.compartments)
        if should_stratify:
            for stratum in strat.strata:
                new_value = comp_values[idx] * strat.comp_split_props[stratum]
                new_comp_values.append(new_value)
        else:
            new_comp_values.append(comp_values[idx])

    return np.array(new_comp_values)

