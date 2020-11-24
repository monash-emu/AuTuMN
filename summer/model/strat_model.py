import copy
import itertools
from typing import List, Dict, Callable, Optional, Union

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

    DEFAULT_DISEASE_STRAIN = "default"
    DEFAULT_MIXING_MATRIX = np.array([[1]])

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

        # A list of the different sub-categories ('strains') of the diease that we are modelling.
        self.disease_strains = [self.DEFAULT_DISEASE_STRAIN]

        # Strata-based multipliers for compartmental infectiousness levels.
        self.infectiousness_levels = {}

        # Compartment-based overwrite values for compartmental infectiousness levels.
        self.individual_infectiousness_adjustments = []

        # Mixing matrices a list of NxN arrays used to calculate force of infection.
        self.mixing_matrices = []

        # A list of dicts that has the strata required to match a row in the mixing matrix.
        self.mixing_categories = [{}]

    def stratify(
        self,
        stratification_name: str,
        strata_request: List[str],
        compartments_to_stratify: List[str],
        comp_split_props: Dict[str, float] = {},
        flow_adjustments: Dict[str, Dict[str, float]] = {},
        infectiousness_adjustments: Dict[str, float] = {},
        mixing_matrix: Union[np.ndarray, Callable[[float], np.ndarray]] = None,
    ):
        """
        Apply a stratification to the model's compartments.

        stratification_name: The name of the stratification
        strata_names: The names of the strata to apply
        compartments_to_stratify: The compartments that will have the stratification applied. Falsey args interpreted as "all".
        comp_split_props: Request to split existing population in the compartments according to specific proportions
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

        # Add this stratification's mixing matrix if a new one is provided.
        if mixing_matrix is not None:
            assert not strat.is_strain(), "Strains cannot have a mixing matrix."
            # Only allow mixing matrix to be supplied if there is a complete stratification.
            assert (
                compartments_to_stratify == self.original_compartment_names
            ), "Mixing matrices only allowed for full stratification."
            self.mixing_matrices.append(mixing_matrix)
            self.mixing_categories = strat.update_mixing_categories(self.mixing_categories)

        # Prepare infectiousness levels for force of infection adjustments.
        self.infectiousness_levels[strat.name] = infectiousness_adjustments

        if strat.is_strain():
            # Track disease strain names, overriding default values.
            self.disease_strains = strat.strata

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

    def add_extra_flow(
        self,
        flow: dict,
        source_strata: dict,
        dest_strata: dict,
        expected_flow_count: int = None,
    ):
        """
        Add a new flow from a set of source compartments to a set of dest compartments.
        This can be done before or after stratification(s).

        The supplied strata will be used to find the source and destination compartments.
        There must be an equal number of source and destination compartments.
        Any unspecified strata will have flows created in parallel.

        flow: The flow to add, same format as when construction the model.
        source_strata: Source strata used to find source compartments.
        dest_strata: Destination strata used to find destination compartments.
        expected_count: (Optional) Expected number of flows to be created.
        """
        assert flow["type"] in Flow.TRANSITION_FLOWS, "Can only add extra transition flows."
        flows_to_add = []
        source_comps = [
            comp for comp in self.compartment_names if comp.is_match(flow["origin"], source_strata)
        ]
        dest_comps = [
            comp for comp in self.compartment_names if comp.is_match(flow["to"], dest_strata)
        ]
        count_dest_comps = len(dest_comps)
        count_source_comps = len(source_comps)
        msg = f"Expected equal number of source and dest compartmentss, but got {count_source_comps} source and {count_dest_comps} dest."
        assert count_dest_comps == count_source_comps, msg
        for source_comp, dest_comp in zip(source_comps, dest_comps):
            flow_to_add = {
                **flow,
                "origin": source_comp.serialize(),
                "to": dest_comp.serialize(),
            }
            flows_to_add.append(flow_to_add)

        if expected_flow_count is not None:
            # Check that we added the expected number of flows.
            actual_flow_count = len(flows_to_add)
            msg = f"Expected to add {expected_flow_count} flows but added {actual_flow_count}"
            assert actual_flow_count == expected_flow_count, msg

        # Add the new flows to the model
        for flow_to_add in flows_to_add:
            self._add_flow(flow_to_add)

        # Update flow compartment idxs for faster lookups.
        self._update_flow_compartment_indices()

    def get_mixing_matrix(self, time: float) -> np.ndarray:
        """
        Returns the final mixing matrix for a given time.
        """
        mixing_matrix = self.DEFAULT_MIXING_MATRIX
        for mm_func in self.mixing_matrices:
            # Assume each mixing matrix is either an np.ndarray or a function of time that returns one.
            mm = mm_func(time) if callable(mm_func) else mm_func
            # Get Kronecker product of old and new mixing matrices.
            mixing_matrix = np.kron(mixing_matrix, mm)

        return mixing_matrix

    def prepare_force_of_infection(self):
        """
        Pre-run calculations to help determine force of infection multiplier at runtime.

        We start with a set of "mixing categories". These categories describe groups of compartments.
        For example, we might have the stratifications age {child, adult} and location {work, home}.
        In this case, the mixing categories would be {child x home, child x work, adult x home, adult x work}.
        Mixing categories are only created when a mixing matrix is supplied during stratification.

        There is a mapping from every compartment to a mixing category.
        This is only true if mixing matrices are supplied only for complete stratifications.
        There are `num_cats` categories and `num_comps` compartments.
        The category matrix is a (num_cats x num_comps) matrix of 0s and 1s, with a 1 when the compartment is in a given category.
        We expect only one category per compartment, but there may be many compartments per category.

        We can multiply the category matrix by the vector of compartment sizes to get the total number of people
        in each mixing category.

        We also create a vector of values in [0, inf) which describes how infectious each compartment is: compartment_infectiousness
        We can use this vector plus the compartment sizes to get the 'effective' number of infectious people per compartment.

        We can use the 'effective infectious' compartment sizes, plus the mixing category matrix
        to get the infected population per mixing category.

        Now that we know:
            - the total population per category
            - the infected population per category
            - the inter-category mixing coefficients (mixing matrix)

        We can calculate the infection density or frequency per category.
        Finally, at runtime, we can lookup which category a given compartment is in and look up its infectious multiplier (density or frequency).
        """
        # Find out the relative infectiousness of each compartment, for each strain.
        # If no strains have been create, we assume a default strain name.
        self.compartment_infectiousness = {
            strain_name: self.get_compartment_infectiousness(strain_name)
            for strain_name in self.disease_strains
        }

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

    def get_compartment_infectiousness(self, strain: str):
        """
        Returns a vector of floats, each representing the relative infectiousness of each compartment.
        If a strain name is provided, find the infectiousness factor *only for that strain*.
        """
        # Figure out which compartments should be infectious
        infectious_mask = np.array(
            [c.has_name_in_list(self.infectious_compartments) for c in self.compartment_names]
        )
        # Find the infectiousness multipliers for each compartment being implemented in the model.
        # Start from assumption that each compartment is not infectious.
        compartment_infectiousness = np.zeros(self.compartment_values.shape)
        # Set all infectious compartments to be equally infectious.
        compartment_infectiousness[infectious_mask] = 1

        # Multiply by used-requested factors for particular strata.
        for idx, compartment in enumerate(self.compartment_names):
            for strat_name, strat_multipliers in self.infectiousness_levels.items():
                for stratum_name, multiplier in strat_multipliers.items():
                    if compartment.has_stratum(strat_name, stratum_name):
                        compartment_infectiousness[idx] *= multiplier

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

                compartment_infectiousness[idx] = value

        if strain != self.DEFAULT_DISEASE_STRAIN:
            # Filter out all values that are not in the given strain.
            strain_mask = np.zeros(self.compartment_values.shape)
            for idx, compartment in enumerate(self.compartment_names):
                if compartment.has_stratum("strain", strain):
                    strain_mask[idx] = 1

            compartment_infectiousness *= strain_mask

        return compartment_infectiousness

    def get_force_idx(self, source: Compartment):
        """
        Returns the index of the source compartment in the infection multiplier vector.
        """
        return self.category_lookup[source.idx]

    def find_infectious_population(self, time: float, compartment_values: np.ndarray):
        """
        Finds the effective infectious population.
        """
        mixing_matrix = self.get_mixing_matrix(time)

        # Calculate total number of people per category.
        # A vector with size (num_cats x 1).
        comp_values_transposed = compartment_values.reshape((compartment_values.shape[0], 1))
        self.category_populations = np.matmul(self.category_matrix, comp_values_transposed)

        # Calculate infectious populations for each strain.
        # Infection density / frequency is the infectious multiplier for each mixing category, calculated for each strain.
        self.infection_density = {}
        self.infection_frequency = {}
        for strain in self.disease_strains:
            strain_compartment_infectiousness = self.compartment_infectiousness[strain]

            # Calculate total infected people per category, including adjustment factors.
            # Returns a vector with size (num_cats x 1).
            infected_values = compartment_values * strain_compartment_infectiousness
            infected_values_transposed = infected_values.reshape((infected_values.shape[0], 1))
            infectious_populations = np.matmul(self.category_matrix, infected_values_transposed)
            self.infection_density[strain] = np.matmul(mixing_matrix, infectious_populations)

            # Calculate total infected person frequency per category, including adjustment factors.
            # A vector with size (num_cats x 1).
            category_frequency = infectious_populations / self.category_populations
            self.infection_frequency[strain] = np.matmul(mixing_matrix, category_frequency)

    def get_infection_frequency_multiplier(self, source: Compartment, dest: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection frequency' calculation.
        """
        idx = self.get_force_idx(source)
        strain = dest._strat_values.get("strain", self.DEFAULT_DISEASE_STRAIN)
        return self.infection_frequency[strain][idx][0]

    def get_infection_density_multiplier(self, source: Compartment, dest: Compartment):
        """
        Get force of infection multiplier for a given compartment,
        using the 'infection density' calculation.
        """
        idx = self.get_force_idx(source)
        strain = dest._strat_values.get("strain", self.DEFAULT_DISEASE_STRAIN)
        return self.infection_density[strain][idx][0]
