from typing import List, Tuple, Dict, Callable

from summer.constants import FlowAdjustment
from summer.stratification import Stratification
from .flow import BaseFlow


class BaseTransitionFlow(BaseFlow):
    """
    A flow where people move from the source compartment, to the destination.
    Eg. infection, recovery, progress of disease.
    """

    def update_compartment_indices(self, mapping: Dict[str, float]):
        """
        Update index which maps flow compartments to compartment value array.
        """
        self.source.idx = mapping[self.source]
        self.dest.idx = mapping[self.dest]

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified flows to replace the current flow.
        """
        is_source_strat = self.source.has_name_in_list(strat.compartments)
        is_dest_strat = self.dest.has_name_in_list(strat.compartments)
        if not (is_dest_strat or is_source_strat):
            # Flow is not stratified, do not stratify this flow.
            return [self]

        new_flows = []
        for stratum in strat.strata:
            # Find new compartments
            if is_source_strat:
                new_source = self.source.stratify(strat.name, stratum)
            else:
                new_source = self.source

            if is_dest_strat:
                new_dest = self.dest.stratify(strat.name, stratum)
            else:
                new_dest = self.dest

            # Find flow adjustments
            # Try find an adjustment for the source compartment.
            adjustment = strat.get_flow_adjustment(self.source, stratum, self.param_name)
            if not adjustment:
                # Otherwise, try find an adjustment for the destination compartment.
                adjustment = strat.get_flow_adjustment(self.dest, stratum, self.param_name)

            if not adjustment and (is_dest_strat and not is_source_strat):
                # If the source is stratified but not the destination, then we need to account
                # for the resulting fan-out of flows by reducing the flow rate.
                num_strata = len(strat.strata)
                entry_fraction = 1.0 / num_strata
                adjustment = (FlowAdjustment.MULTIPLY, entry_fraction)

            if adjustment:
                new_adjustments = [*self.adjustments, adjustment]
            else:
                new_adjustments = self.adjustments

            new_flow = self.copy(
                source=new_source,
                dest=new_dest,
                param_name=self.param_name,
                param_func=self.param_func,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows
