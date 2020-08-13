from typing import List, Tuple, Dict, Callable

from summer.stratification import Stratification
from .flow import BaseFlow


class BaseExitFlow(BaseFlow):
    """
    A flow where people exit the source compartment, but there is no destination
    Eg. deaths, exportation
    """

    def update_compartment_indices(self, mapping: Dict[str, float]):
        """
        Update index which maps flow compartments to compartment value array.
        """
        self.source.idx = mapping[self.source]

    def stratify(self, strat: Stratification) -> List[BaseFlow]:
        """
        Returns a list of new, stratified exit flows to replace the current flow.
        """
        if not self.source.has_name_in_list(strat.compartments):
            # Flow source is not stratified, do not stratify this flow.
            return [self]

        new_flows = []
        for stratum in strat.strata:
            adjustment = strat.get_flow_adjustment(self.source, stratum, self.param_name)
            if adjustment:
                new_adjustments = [*self.adjustments, adjustment]
            else:
                new_adjustments = self.adjustments

            new_source = self.source.stratify(strat.name, stratum)
            new_flow = self.copy(
                source=new_source,
                param_name=self.param_name,
                param_func=self.param_func,
                adjustments=new_adjustments,
            )
            new_flows.append(new_flow)

        return new_flows
