import copy
import itertools
from typing import List, Dict, Callable

import numpy as np

from summer.constants import (
    Flow,
    BirthApproach,
    IntegrationType,
)
from summer.model.epi_model import EpiModel
from summer.model.utils.validation import validate_stratify
from summer.compartment import Compartment
from summer.flow import AgeingFlow
from summer.stratification import (
    Stratification,
    get_stratified_compartment_names,
    get_stratified_compartment_values,
)


class StratifiedModel(EpiModel):
    """
    Stratified compartmental model. 
    """

    def __init__(
        self,
        times: List[float],
        compartment_names: List[str],
        initial_conditions: Dict[str, float],
        parameters: Dict[str, float],
        requested_flows: List[dict],
        infectious_compartments: List[str],
        birth_approach: str,
        entry_compartment: str,
        starting_population: int,
    ):
        super().__init__(
            times=times,
            compartment_names=compartment_names,
            initial_conditions=initial_conditions,
            parameters=parameters,
            requested_flows=requested_flows,
            infectious_compartments=infectious_compartments,
            birth_approach=birth_approach,
            entry_compartment=entry_compartment,
            starting_population=starting_population,
        )

        # Keeps track of Stratifications that have been applied.
        self.stratifications = []
        # Keeps track of original, pre-stratified compartment names.
        self.original_compartment_names = [Compartment.deserialize(n) for n in compartment_names]

        # Strata-based multipliers for compartmental infectiousness levels.
        self.infectiousness_levels = {}
        # Compartment-based overwrite values for compartmental infectiousness levels.
        self.individual_infectiousness_adjustments = []

        # Mixing matrix, NxN array used to calculate force of infection.
        # Columns are the strata who are infectors.
        # Rows are the strata who are infected.
        # This matrix can be stratified multiple times.
        self._static_mixing_matrix = np.array([[1]])
        # A time-based function that can replace the mixing matrix.
        # This cannot be automatically stratified.
        self._dynamic_mixing_matrix = None
        # A list of dicts that has the strata required to match a row in the mixing matrix.
        self.mixing_categories = [{}]

    def set_dynamic_mixing_matrix(self, matrix_func: Callable[[float], np.ndarray]):
        """
        Set model to use a time-varying mixing matrix.
        FIXME: This API is very very unintuitive. It's not obvious which mixing matrix gets used by the model.
        """
        self._dynamic_mixing_matrix = matrix_func

    def stratify(
        self,
        stratification_name: str,
        strata_request: List[str],
        compartments_to_stratify: List[str],
        comp_split_props: Dict[str, float] = {},
        flow_adjustments: Dict[str, Dict[str, float]] = {},
        infectiousness_adjustments: Dict[str, float] = {},
        mixing_matrix: np.ndarray = None,
    ):
        """
        Apply a stratification to the model's compartments.

        stratification_name: The name of the stratification
        strata_names: The names of the strata to apply
        compartments_to_stratify: The compartments that will have the stratification applied. Falsey args interpreted as "all".
        comp_split_props: Request to split existing population in the compartments according to specific proportions
        flow_adjustments: TODO
        infectiousness_adjustments: TODO
        mixing_matrix: TODO
        """
        validate_stratify(
            self,
            stratification_name,
            strata_request,
            compartments_to_stratify,
            comp_split_props,
            flow_adjustments,
            infectiousness_adjustments,
            mixing_matrix,
        )
        strat = Stratification(
            name=stratification_name,
            strata=strata_request,
            compartments=compartments_to_stratify,
            comp_split_props=comp_split_props,
            flow_adjustments=flow_adjustments,
        )
        self.stratifications.append(strat)

        # Stratify the mixing matrix if a new one is provided.
        if mixing_matrix is not None:
            # Use Kronecker product of old and new mixing matrices.
            self._static_mixing_matrix = np.kron(self._static_mixing_matrix, mixing_matrix)
            self.mixing_categories = strat.update_mixing_categories(self.mixing_categories)

        # Prepare infectiousness levels for force of infection adjustments.
        self.infectiousness_levels[strat.name] = infectiousness_adjustments

        # Stratify compartments, split according to split_proportions
        prev_compartment_names = copy.copy(self.compartment_names)
        self.compartment_names = get_stratified_compartment_names(strat, self.compartment_names)
        self.compartment_values = get_stratified_compartment_values(
            strat, prev_compartment_names, self.compartment_values
        )
        for idx, c in enumerate(self.compartment_names):
            c.idx = idx

        # Stratify flows
        prev_flows = self.flows
        self.flows = []
        for flow in prev_flows:
            self.flows += flow.stratify(strat)

        if strat.is_ageing():
            ageing_flows, ageing_params = AgeingFlow.create(
                strat, prev_compartment_names, self.get_parameter_value
            )
            self.flows += ageing_flows
            self.parameters.update(ageing_params)

        # Update indicies used by flows to lookup compartment values.
        compartment_idx_lookup = {name: idx for idx, name in enumerate(self.compartment_names)}
        for flow in self.flows:
            flow.update_compartment_indices(compartment_idx_lookup)

    def prepare_force_of_infection(self):
        """
        Pre-run calculations to help determine force of infection multiplier at runtime. 
        """
        # Figure out which compartments should be infectious
        infectious_mask = np.array(
            [c.has_name_in_list(self.infectious_compartments) for c in self.compartment_names]
        )
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        # Start from assumption that each compartment is not infectious.
        self.infectiousness_multipliers = np.zeros(self.compartment_values.shape)
        # Set all infectious compartments to be equally infectious.
        self.infectiousness_multipliers[infectious_mask] = 1

        # Multiply by used-requested factors for particular strata.
        for idx, compartment in enumerate(self.compartment_names):
            for strat_name, strat_multipliers in self.infectiousness_levels.items():
                for stratum_name, multiplier in strat_multipliers.items():
                    if compartment.has_stratum(strat_name, stratum_name):
                        self.infectiousness_multipliers[idx] *= multiplier

        # Override with user-requested values for particular compartments.
        # FIXME: This is yucky and requires user to hack on model post/pre stratification.
        for adj in self.individual_infectiousness_adjustments:
            comp_name = adj["comp_name"]
            comp_strata = adj["comp_strata"]
            value = adj["value"]
            for idx, comp in enumerate(self.compartment_names):
                if not comp.has_name(comp_name):
                    continue

                if not all(comp.has_stratum(k, v) for k, v in comp_strata.items()):
                    continue

                self.infectiousness_multipliers[idx] = value

        # Create a matrix that tracks which categories each compartment is in.
        # A matrix with size (num_cats x num_comps).
        num_comps = len(self.compartment_names)
        num_categories = len(self.mixing_categories)
        self.category_lookup = {}  # Map compartments to categories.
        self.category_matrix = np.zeros((num_categories, num_comps))
        for i, category in enumerate(self.mixing_categories):
            for j, comp in enumerate(self.compartment_names):
                if all(comp.has_stratum(k, v) for k, v in category.items()):
                    self.category_matrix[i][j] = 1
                    self.category_lookup[j] = i

    def get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self.category_lookup[source.idx]

    def find_infectious_population(self, time: float, compartment_values: np.ndarray):
        """
        Finds the effective infectious population.
        """
        if self._dynamic_mixing_matrix:
            mixing_matrix = self._dynamic_mixing_matrix(time)
        else:
            mixing_matrix = self._static_mixing_matrix

        # Calculate total number of people per category.
        # A vector with size (num_cats x 1).
        comp_values_transposed = compartment_values.reshape((compartment_values.shape[0], 1))
        self.category_populations = np.matmul(self.category_matrix, comp_values_transposed)

        # Calculate total infected people per category, including adjustment factors.
        # Returns a vector with size (num_cats x 1).
        infected_values = compartment_values * self.infectiousness_multipliers
        infected_values_transposed = infected_values.reshape((infected_values.shape[0], 1))
        infectious_populations = np.matmul(self.category_matrix, infected_values_transposed)
        self.infection_density = np.matmul(mixing_matrix, infectious_populations)

        # Calculate total infected person frequency per category, including adjustment factors.
        # A vector with size (num_cats x 1).
        category_frequency = infectious_populations / self.category_populations
        self.infection_frequency = np.matmul(mixing_matrix, category_frequency)

    def get_infection_frequency_multipier(self, source: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection frequency' calculation.
        """
        idx = self.get_force_idx(source)
        return self.infection_frequency[idx][0]

    def get_infection_density_multipier(self, source: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection density' calculation.
        """
        idx = self.get_force_idx(source)
        return self.infection_density[idx][0]
