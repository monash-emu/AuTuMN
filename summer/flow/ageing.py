from typing import List, Tuple, Dict, Callable

from summer.constants import Flow as FlowType
from summer.compartment import Compartment
from summer.stratification import Stratification

from .standard import StandardFlow


class AgeingFlow(StandardFlow):
    def __repr__(self):
        return f"<AgeingFlow from {self.source} to {self.dest} with {self.param_name}>"

    @staticmethod
    def create(
        strat: Stratification,
        compartments: List[Compartment],
        param_func: Callable[[str, float], float],
    ):
        """
        Create inter-compartmental flows for ageing from one stratum to the next.
        The ageing rate is proportional to the width of the age bracket.
        It's assumed that both ages and model timesteps are in years.
        """
        assert strat.is_ageing()
        ageing_flows = []
        ageing_params = {}
        ages = list(sorted(map(int, strat.strata)))
        for age_idx in range(len(ages) - 1):
            start_age = int(ages[age_idx])
            end_age = int(ages[age_idx + 1])
            param_name = f"ageing{start_age}to{end_age}"
            ageing_rate = 1.0 / (end_age - start_age)
            ageing_params[param_name] = ageing_rate
            for comp in compartments:
                if not comp.has_name_in_list(strat.compartments):
                    # Don't include unstratified compartments
                    continue

                flow = AgeingFlow(
                    source=comp.stratify(strat.name, str(start_age)),
                    dest=comp.stratify(strat.name, str(end_age)),
                    param_name=param_name,
                    param_func=param_func,
                )
                ageing_flows.append(flow)

        return ageing_flows, ageing_params
