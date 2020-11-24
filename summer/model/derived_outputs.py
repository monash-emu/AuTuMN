from typing import List, Dict, Callable

import numpy as np
from numpy.lib.function_base import append

from summer.constants import Flow


class DerivedOutputCalculator:
    def __init__(self):
        """
        Calculates derived outputs from a set of outputs
        """
        self._flow_outputs = {}
        self._function_outputs = []

    def add_flow_derived_outputs(self, flow_outputs: dict):
        self._flow_outputs.update(flow_outputs)

    def add_function_derived_outputs(self, func_outputs: dict):
        for k, v in func_outputs.items():
            self._function_outputs.append([k, v])

    def calculate(self, model):
        """
        Returns the derived outputs for a model.
        """
        derived_outputs = {}

        # Initialize outputs
        base_output = np.zeros(model.times.shape)

        # Initialize flow outputs
        flow_lookup = {}
        for output_name, flow_output in self._flow_outputs.items():
            flow_lookup[output_name] = flow_output.filter_flows(model.flows)
            derived_outputs[output_name] = base_output.copy()

        # Initialize function outputs
        for output, _ in self._function_outputs:
            derived_outputs[output] = base_output.copy()

        # Calculate derived outputs for all model time steps.
        for time_idx, time in enumerate(model.times):
            compartment_values = model.restore_past_state(time_idx)
            # Calculate derived flow outputs.
            for output_name, output_flows in flow_lookup.items():
                for flow in output_flows:
                    net_flow = flow.get_net_flow(compartment_values, time)
                    derived_outputs[output_name][time_idx] += net_flow

            # Calculate derived function outputs
            for output_name, func in self._function_outputs:
                derived_outputs[output_name][time_idx] = func(
                    time_idx, model, compartment_values, derived_outputs
                )

        return derived_outputs


class InfectionDeathFlowOutput:
    """
    A request to track a set of infection death flow in the model's derived outputs.
    """

    def __init__(
        self,
        source: str,
        source_strata: Dict[str, str],
    ):
        self.source = source
        self.source_strata = source_strata

    def filter_flows(self, flows):
        filtered_flows = []
        for flow in flows:
            if flow.type != Flow.DEATH:
                continue

            if not flow.source.has_name(self.source):
                continue

            # Check source strata
            if self.source_strata:
                is_source_strata_valid = all(
                    flow.source.has_stratum(k, v) for k, v in self.source_strata.items()
                )
                if not is_source_strata_valid:
                    continue

            filtered_flows.append(flow)

        return filtered_flows


class TransitionFlowOutput:
    """
    A request to track a set of transition flows in the model's derived outputs.
    """

    def __init__(
        self,
        source: str,
        dest: str,
        source_strata: Dict[str, str],
        dest_strata: Dict[str, str],
    ):
        self.source = source
        self.dest = dest
        self.source_strata = source_strata
        self.dest_strata = dest_strata

    def filter_flows(self, flows):
        filtered_flows = []
        for flow in flows:
            if flow.type not in Flow.TRANSITION_FLOWS:
                continue

            if not flow.source.has_name(self.source):
                continue

            if not flow.dest.has_name(self.dest):
                continue

            # Check source strata
            if self.source_strata:
                is_source_strata_valid = all(
                    flow.source.has_stratum(k, v) for k, v in self.source_strata.items()
                )
                if not is_source_strata_valid:
                    continue

            # Check dest strata
            if self.dest_strata:
                is_dest_strata_valid = all(
                    flow.dest.has_stratum(k, v) for k, v in self.dest_strata.items()
                )
                if not is_dest_strata_valid:
                    continue

            filtered_flows.append(flow)

        return filtered_flows
