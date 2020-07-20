from typing import List, Tuple, Dict, Callable

from summer.stratification import Stratification
from summer.constants import FlowAdjustment

from .flow import BaseFlow


class BaseEntryFlow(BaseFlow):
    """
    A flow where people enter the destination compartment, but there is no source.
    Eg. births, importation.
    """

    def update_compartment_indices(self, mapping: Dict[str, float]):
        """
        Update index which maps flow compartments to compartment value array.
        """
        self.dest.idx = mapping[self.dest]

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified entry flows to replace the current flow.
        """
        if not self.dest.has_name_in_list(strat.compartments):
            # Flow destination is not stratified, do not stratify this flow.
            return [self]

        new_flows = []
        for stratum in strat.strata:
            adjustment = None
            if strat.is_ageing():
                # Use special rules for ageing.
                if stratum == "0":
                    # Babies get born at age 0, add a null op to prevent default behaviour.
                    adjustment = (FlowAdjustment.MULTIPLY, 1)
                else:
                    # Babies do not get born at any other age.
                    adjustment = (FlowAdjustment.OVERWRITE, 0)
            else:
                # Not an ageing stratification, check for user-specified flow adjustments.
                adjustment = strat.get_flow_adjustment(self.dest, stratum, self.param_name)

            if not adjustment:
                # Default to equally dividing entry population between all strata.
                num_strata = len(strat.strata)
                entry_fraction = 1.0 / num_strata
                adjustment = (FlowAdjustment.MULTIPLY, entry_fraction)

            new_adjustments = [*self.adjustments, adjustment]
            new_dest = self.dest.stratify(strat.name, stratum)
            new_flow = self.copy(
                dest=new_dest,
                param_name=self.param_name,
                param_func=self.param_func,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows
