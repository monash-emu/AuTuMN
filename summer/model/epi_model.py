import copy
import logging
from typing import List, Dict, Tuple, Callable
from functools import lru_cache

import matplotlib.pyplot
import numpy as np
import pandas as pd

from summer.constants import (
    Flow,
    BirthApproach,
    IntegrationType,
)
from summer.compartment import Compartment
from summer.flow import (
    CrudeBirthFlow,
    InfectionDensityFlow,
    InfectionFrequencyFlow,
    ReplacementBirthFlow,
    StandardFlow,
    InfectionDeathFlow,
    UniversalDeathFlow,
    ImportFlow,
)

from .utils.solver import solve_ode
from .utils.validation import validate_model
from .derived_outputs import DerivedOutputCalculator


logger = logging.getLogger(__name__)

OUTPUTS_NEGATIVE_TOLERANCE = -1e-4


class EpiModel:
    """
    Constructs and runs compartment-based epidemiological models, 
    typically of infectious disease transmission.

    Compartment attributes
    compartment_names: The model compartments.
    compartment_values: The compartment sizes.
    entry_compartment: The compartment to add new population members to.
    initial_conditions: Proprotion of starting population for each compartment.
    starting_population: Total starting population.

    Flow attributes
    flows:  The flows between compartments
    parameters: Weights for each flow.
    time_variants: Time-varying flow weights.
    birth_approach: How we handle births.
    
    Runtime attributes
    times: Time steps at which outputs are to be evaluated.
    outputs: All the evaluated compartment sizes over requested times.
    total_deaths: Tracks total deaths for the 'replace deaths' birth approach.
    infectious_compartment: Names of the infectious compartments for infection flows
    populations_infectious: The current size of the infectious population.
    total_population: The current size of the total population, relevant for frequency-dependent transmission.
    infectious_mask: Numpy mask for identifying the infectious compartments.
    """

    def __init__(
        self,
        times: List[float],
        compartment_names: List[str],
        initial_conditions: Dict[str, float],
        parameters: Dict[str, float],
        requested_flows: List[dict],
        birth_approach: str,
        starting_population: int,
        infectious_compartments: List[str],
        entry_compartment: str,
    ):
        model_kwargs = {
            "times": times,
            "compartment_names": compartment_names,
            "initial_conditions": initial_conditions,
            "parameters": parameters,
            "requested_flows": requested_flows,
            "birth_approach": birth_approach,
            "starting_population": starting_population,
            "infectious_compartments": infectious_compartments,
            "entry_compartment": entry_compartment,
        }
        validate_model(model_kwargs)
        # Flows
        self.flows = []
        self.time_variants = {}
        self.parameters = {}

        # Compartments
        self.compartment_names = [Compartment(n) for n in compartment_names]
        self.infectious_compartments = [Compartment(n) for n in infectious_compartments]

        # Derived outputs
        self.derived_outputs = {}
        self._derived_calc = DerivedOutputCalculator()

        self.times = times

        # Populate model compartments with the values set in `initial conditions`.
        self.compartment_values = np.zeros(len(compartment_names))
        pop_remainder = starting_population - sum(initial_conditions.values())
        for idx, comp_name in enumerate(compartment_names):
            if comp_name in initial_conditions:
                self.compartment_values[idx] = initial_conditions[comp_name]

            if comp_name == entry_compartment:
                self.compartment_values[idx] += pop_remainder

        self.birth_approach = birth_approach
        self.parameters = parameters
        self.parameters["crude_birth_rate"] = self.parameters.get("crude_birth_rate", 0)
        self.parameters["universal_death_rate"] = self.parameters.get("universal_death_rate", 0)

        # Add user-specified flows
        _entry_comp = Compartment.deserialize(entry_compartment)
        self.flows = []
        for requested_flow in requested_flows:
            flow = None
            f_source = requested_flow.get("origin")
            f_source = Compartment.deserialize(f_source) if f_source else None
            f_dest = requested_flow.get("to")
            f_dest = Compartment.deserialize(f_dest) if f_dest else None
            f_param = requested_flow["parameter"]
            f_type = requested_flow["type"]
            if f_type == Flow.STANDARD:
                flow = StandardFlow(
                    source=f_source,
                    dest=f_dest,
                    param_name=f_param,
                    param_func=self.get_parameter_value,
                )
            elif f_type == Flow.INFECTION_FREQUENCY:
                flow = InfectionFrequencyFlow(
                    source=f_source,
                    dest=f_dest,
                    param_name=f_param,
                    param_func=self.get_parameter_value,
                    find_infectious_multiplier=self.get_infection_frequency_multipier,
                )
            elif f_type == Flow.INFECTION_DENSITY:
                flow = InfectionDensityFlow(
                    source=f_source,
                    dest=f_dest,
                    param_name=f_param,
                    param_func=self.get_parameter_value,
                    find_infectious_multiplier=self.get_infection_density_multipier,
                )
            elif f_type == Flow.DEATH:
                flow = InfectionDeathFlow(
                    source=f_source, param_name=f_param, param_func=self.get_parameter_value
                )
            elif f_type == Flow.IMPORT:
                flow = ImportFlow(
                    dest=_entry_comp, param_name=f_param, param_func=self.get_parameter_value,
                )

            if flow:
                self.flows.append(flow)

        # Add birth flows.
        if self.birth_approach == BirthApproach.ADD_CRUDE:
            param_name = "crude_birth_rate"
            flow = CrudeBirthFlow(
                dest=_entry_comp, param_name=param_name, param_func=self.get_parameter_value,
            )
            self.flows.append(flow)
        elif self.birth_approach == BirthApproach.REPLACE_DEATHS:
            self.total_deaths = 0
            flow = ReplacementBirthFlow(dest=_entry_comp, get_total_deaths=self.get_total_deaths)
            self.flows.append(flow)

        # Add non-disease death flows
        param_name = "universal_death_rate"
        for compartment_name in self.compartment_names:
            flow = UniversalDeathFlow(
                source=compartment_name, param_name=param_name, param_func=self.get_parameter_value,
            )
            self.flows.append(flow)

        # Create lookup table for quickly getting / setting compartments by name
        compartment_idx_lookup = {name: idx for idx, name in enumerate(self.compartment_names)}
        for flow in self.flows:
            flow.update_compartment_indices(compartment_idx_lookup)

        for idx, c in enumerate(self.compartment_names):
            c.idx = idx

    # Cache return values to prevent re-computation. This will a little leak memory, which is fine.
    # Floating point return type is 8 bytes, meaning 2**17 values is ~1MB of memory.
    @lru_cache(maxsize=2 ** 17)
    def get_parameter_value(self, name: str, time: float):
        """
        Get parameter value at a given time
        """
        if name in self.time_variants:
            return self.time_variants[name](time)
        else:
            return self.parameters[name]

    def prepare_to_run(self):
        """
        Pre-run setup.
        """
        # Split flows up into 3 groups: entry, exit and transition.
        self.entry_flows = [f for f in self.flows if f.type in Flow.ENTRY_FLOWS]
        self.exit_flows = [f for f in self.flows if f.type in Flow.EXIT_FLOWS]
        self.transition_flows = [f for f in self.flows if f.type in Flow.TRANSITION_FLOWS]

        # Check we didn't miss any flows
        split_flows = [self.entry_flows, self.exit_flows, self.transition_flows]
        num_split_flows = sum(len(fs) for fs in split_flows)
        assert len(self.flows) == num_split_flows

        # Determine order of how flows are run.
        # We apply deaths before births so that we can use 'total deaths' to calculate the birth rate, if required.
        self.flow_functions = [
            self.apply_transition_flows,
            self.apply_exit_flows,
            self.apply_entry_flows,
        ]
        self.prepare_force_of_infection()

    def prepare_force_of_infection(self):
        """
        Pre-run calculations to help determine force of infection multiplier at runtime. 
        """
        # Figure out which compartments should be infectious
        self.infectious_mask = [
            c.has_name_in_list(self.infectious_compartments) for c in self.compartment_names
        ]

    def run_model(self, integration_type=IntegrationType.SOLVE_IVP, solver_args={}):
        """
        Calculates the model's outputs using an ODE solver.

        The ODE is solved over the user-specified timesteps (self.times),
        using the user-specified initial conditions (self.compartment_values).

        The final result is an array of compartment values at each timestep (self.outputs).
        Also calculates post-processing outputs after the ODE integration is complete.
        """
        self.prepare_to_run()
        self.outputs = solve_ode(
            integration_type, self.get_flow_rates, self.compartment_values, self.times, solver_args
        )

        if np.any(self.outputs < OUTPUTS_NEGATIVE_TOLERANCE):
            msg = f"Negative compartment size found in model output, value smaller than {OUTPUTS_NEGATIVE_TOLERANCE}"
            raise ValueError(msg)

        self.derived_outputs = self.calculate_derived_outputs()

    def get_flow_rates(self, compartment_values: np.ndarray, time: float):
        """
        Get net flows into, out of, and between all compartments.
        Order of args determined by solve_ode func.
        """
        # Zero out compartment sizes to prevent negative values from messing up calcs.
        comp_vals = compartment_values.copy()
        zero_mask = comp_vals < 0
        comp_vals[zero_mask] = 0
        self.prepare_time_step(time, comp_vals)
        flow_rates = np.zeros(comp_vals.shape)
        for flow_func in self.flow_functions:
            flow_rates = flow_func(flow_rates, comp_vals, time)

        return flow_rates

    def prepare_time_step(self, time: float, compartment_values: np.ndarray):
        self.total_deaths = 0
        self.find_infectious_population(time, compartment_values)

    def apply_transition_flows(
        self, flow_rates: np.ndarray, compartment_values: np.ndarray, time: float
    ):
        """
        Apply fixed or infection-related inter-compartmental transition flows.
        """
        for flow in self.transition_flows:
            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.source.idx] -= net_flow
            flow_rates[flow.dest.idx] += net_flow

        return flow_rates

    def apply_exit_flows(self, flow_rates: np.ndarray, compartment_values: np.ndarray, time: float):
        """
        Apply exit flows: deaths, exports, etc.
        """
        for flow in self.exit_flows:
            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.source.idx] -= net_flow
            if flow.type in Flow.DEATH_FLOWS:
                self.total_deaths += net_flow

        return flow_rates

    def apply_entry_flows(
        self, flow_rates: np.ndarray, compartment_values: np.ndarray, time: float
    ):
        """
        Apply entry flows: births, imports, etc.
        """
        for flow in self.entry_flows:

            net_flow = flow.get_net_flow(compartment_values, time)
            flow_rates[flow.dest.idx] += net_flow

        return flow_rates

    def get_total_deaths(self):
        return self.total_deaths

    def get_infection_frequency_multipier(self, source: Compartment):
        return self.population_infectious / self.population_total

    def get_infection_density_multipier(self, source: Compartment):
        return self.population_infectious

    def find_infectious_population(self, time: float, compartment_values: np.ndarray):
        """
        Finds the effective infectious population
        """
        self.population_infectious = sum(compartment_values[self.infectious_mask])
        self.population_total = sum(compartment_values)

    def restore_past_state(self, time_idx: int):
        """
        Update the model's tracked quantities to a particular time step.
        This is used for calculating derived outputs.
        Returns the compartment values for that time step.
        """
        time = self.times[time_idx]
        compartment_values = self.outputs[time_idx]
        self.prepare_time_step(time, compartment_values)
        return compartment_values

    def calculate_derived_outputs(self):
        return self._derived_calc.calculate(self)

    def add_flow_derived_outputs(self, flow_outputs: dict):
        self._derived_calc.add_flow_derived_outputs(flow_outputs)

    def add_function_derived_outputs(self, func_outputs):
        self._derived_calc.add_function_derived_outputs(func_outputs)
