import copy
import itertools
from functools import lru_cache
from typing import List, Dict

import numpy

from summer.constants import Compartment, Flow, BirthApproach, Stratification, IntegrationType
from .epi_model import EpiModel
from .utils import (
    convert_boolean_list_to_indices,
    create_cumulative_dict,
    create_function_of_function,
    create_multiplicative_function,
    create_stratified_name,
    create_stratum_name,
    create_time_variant_multiplicative_function,
    element_list_multiplication,
    element_list_division,
    extract_reversed_x_positions,
    find_name_components,
    find_stem,
    increment_list_by_index,
)

STRATA_EQUILIBRATION_FACTOR = 0.01
OVERWRITE_CHARACTER = "W"
OVERWRITE_KEY = "overwrite"


class StratifiedModel(EpiModel):
    """
    stratified version of the epidemiological model that inherits from EpiModel above, which is a concrete class and
        could in theory run stratified models independently
    however, this class should make the stratification process more algorithmic, easier and more reliable

    :attribute adaptation_functions: dict
        single stage functions representing each stratified parameter component, from which to build the final functions
            (i.e. final_parameter_functions)
    :attribute all_stratifications: dictionary
        keys are all the stratification names implemented so far. values are the list of strata for each stratification
    :attribute available_death_rates: list
        single strata names for which population_wide mortality will be adjusted (or over-written)
    :attribute compartment_types_to_stratify: list
        the compartments that are being stratified at this round of model stratification
    :attribute final_parameter_functions: dict
        a function representing each parameter that will be implemented during integration,
            constructed recursively for stratification
    :attribute full_stratifications_list: list
        all the stratification names implemented so far that apply to all of the compartment types
    :attribute heterogeneous_mixing: bool
        whether any stratification has requested heterogeneous mixing, such that it will be implemented
    :attribute infectious_compartments: tuple
        all of the compartment stems that represent compartments with some degree of infectiousness
    :attribute infectious_indices: dict
        keys are strains being implemented with "all_strains" an additional standard key, such that models that are not
            stratified by strain will only have the key "all_strains"
        values are lists of the indices of the compartments that are infectious for that strain (or overall)
    :attribute infectious_denominators: float
        total size of the population, which effective infectious population will be divided through by in the case of
            frequency-dependent transmission
    :attribute infectious_populations: dict
        keys are strains
        values are lists with each list element representing a mixing category, so that this can be multiplied through
            by a row of the mixing matrix
    :attribute infectiousness_adjustments: dict
        user-submitted adjustments to infectiousness for the stratification currently being implemented
    :attribute infectiousness_levels: dict
        keys are any strata for any stratification for which infectiousness will be adjusted, which does not need to be
            exhaustive
        values are their relative multipliers
    :attribute infectiousness_multipliers: list
        multipliers for the relative infectiousness of each compartment attributable to stratification, regardless of
            whether they are actually infectious compartments or not and with arbitrary values which start from one and
            are then modified by the user requests
    :attribute mixing_categories: list
        the effective mixing categories, which consists of all the possible combinations of all the strata within the
            model's full stratifications that incorporate heterogeneous mixing
        contents are strings joined with the standard linking character
    :attribute mixing_denominator_indices: dict
        keys are te mixing categories
        values are lists of the indices that should be used to calculate the infectious population for that mixing
            category
    :attribute mixing_matrix: numpy array
        array formed by taking the kronecker product of all the mixing matrices provided for full stratifications for
            which heterogeneous mixing was requested
    :attribute mortality_components: dict
        keys for the name of each compartment, values the list of functions needed to recursively create the functions
            to calculate the mortality rates for each compartment
    :attribute overwrite_character: str
        standard string (usually single character and currently "W") to indicate that a stratum request is intended to
            over-write less stratified parameters
    :attribute overwrite_key: str
        standard string used by model to identify the dictionary element that represents the over-write parameters,
            rather than a request to a particular stratum
    :attribute overwrite_parameters: list
        parameters which will result in all the less stratified parameters closer to the stratification tree's trunk
            being ignored
    :attribute parameter_components: dict
        keys for the name of each transition parameter, values the list of functions needed to recursively create the
            functions to create these parameter values
    :attribute parameters: dict
        same format as for EpiModel (but described here again given the other parameter-related attributes)
        unprocessed parameters, which may be either float values or strings pointing to the keys of time_variants
    :attribute removed_compartments: list
        all unstratified compartments that have been removed through the stratification process
    :attribute overwrite_parameters: list
        any parameters that are intended as absolute values to be applied to that stratum and not multipliers for the
            unstratified parameter further up the tree
    :attribute strain_mixing_elements: dict
        first tier of keys is strains
        second tier of keys is mixing categories
        content of lists at lowest/third tier is the indices of the compartments that are relevant to this strain and
            category
    :attribute strain_mixing_multipliers: dict
        first tier of keys is strains
        second tier of keys is mixing categories
        content of lists at lowest/third tier is the final infectiousness multiplier for the compartments for this
            strain and category
    :attribute strains: list
        the strata to the strains stratification with specific behaviour
    """

    """
    general methods
    """

    def add_compartment(self, new_compartment_name, new_compartment_value):
        """
        add a compartment by specifying its name and the starting value for it to take

        :param new_compartment_name: str
            name of the new compartment to be created
        :param new_compartment_value: float
            initial value to be assigned to the new compartment before integration
        """
        self.compartment_names.append(new_compartment_name)
        self.compartment_values.append(new_compartment_value)
        self.output_to_user("adding compartment: %s" % new_compartment_name)

    def remove_compartment(self, compartment_name):
        """
        remove a compartment by taking the element out of the compartment_names and compartment_values attributes
        store name of removed compartment in a separate attribute

        :param compartment_name: str
            name of compartment to be removed
        """
        self.removed_compartments.append(compartment_name)
        del self.compartment_values[self.compartment_names.index(compartment_name)]
        del self.compartment_names[self.compartment_names.index(compartment_name)]
        self.output_to_user("removing compartment: %s" % compartment_name)

    def __init__(
        self,
        times,
        compartment_types,
        initial_conditions,
        parameters,
        requested_flows,
        infectious_compartment=(Compartment.EARLY_INFECTIOUS,),
        birth_approach=BirthApproach.NO_BIRTH,
        verbose=False,
        reporting_sigfigs=4,
        entry_compartment=Compartment.SUSCEPTIBLE,
        starting_population=1,
        output_connections=None,
        death_output_categories=None,
        derived_output_functions=None,
        ticker=False,
    ):
        super().__init__(
            times,
            compartment_types,
            initial_conditions,
            parameters,
            requested_flows,
            infectious_compartment,
            birth_approach,
            verbose,
            reporting_sigfigs,
            entry_compartment,
            starting_population,
            output_connections,
            death_output_categories,
            derived_output_functions,
            ticker,
        )
        self.full_stratification_list = []
        self.removed_compartments = []
        self.overwrite_parameters = []
        self.compartment_types_to_stratify = []
        self.infectious_denominators = []
        self.strains = []
        self.mixing_categories = []
        self.unstratified_compartment_names = []
        self.all_stratifications = {}
        self.infectiousness_adjustments = {}
        self.final_parameter_functions = {}
        self.adaptation_functions = {}
        self.infectiousness_levels = {}
        self.infectious_indices = {}
        self.infectious_compartments = {}
        self.infectiousness_multipliers = {}
        self.parameter_components = {}
        self.mortality_components = {}
        self.infectious_populations = {}
        self.strain_mixing_elements = {}
        self.strain_mixing_multipliers = {}
        self.strata_indices = {}
        self.target_props = {}
        self.cumulative_target_props = {}
        self.individual_infectiousness_adjustments = []
        self.heterogeneous_mixing = False
        self.mixing_matrix = None
        self.available_death_rates = [""]
        self.dynamic_mixing_matrix = False
        self.mixing_indices = {}
        self.infectious_denominator_shadow = []

    """
    stratification methods
    """

    def stratify(
        self,
        stratification_name,
        strata_request,
        compartment_types_to_stratify,
        requested_proportions,
        entry_proportions={},
        adjustment_requests=(),
        infectiousness_adjustments={},
        mixing_matrix=None,
        target_props=None,
        verbose=True,
    ):
        """
        calls to initial preparation, checks and methods that stratify the various aspects of the model

        :param stratification_name:
            see prepare_and_check_stratification
        :param strata_request:
            see find_strata_names_from_input
        :param compartment_types_to_stratify:
            see check_compartment_request
        :param adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        :param requested_proportions:
            see prepare_starting_proportions
        :param entry_proportions:

        :param infectiousness_adjustments:

        :param mixing_matrix:
            see check_mixing
        :param verbose: bool
            whether to report on progress
            note that this can be changed at this stage from what was requested at the original unstratified model
                construction
        :param target_props: dict
            keys are the strata being implemented at this call to stratify
            values are the desired proportions to target
        """

        # check inputs correctly specified
        strata_names, adjustment_requests = self.prepare_and_check_stratification(
            stratification_name,
            strata_request,
            compartment_types_to_stratify,
            adjustment_requests,
            target_props,
            verbose,
        )

        # work out ageing flows - comes first, so that the compartment names remain in the unstratified form
        if stratification_name == "age":
            self.set_ageing_rates(strata_names)

        # retain copy of compartment names in their stratified form to refer back to during stratification process
        self.unstratified_compartment_names = copy.copy(self.compartment_names)

        # stratify the compartments
        requested_proportions = self.prepare_starting_proportions(
            strata_names, requested_proportions
        )
        self.stratify_compartments(
            stratification_name,
            strata_names,
            requested_proportions,
            self.compartment_types_to_stratify,
        )

        # stratify the flows
        self.stratify_transition_flows(
            stratification_name,
            strata_names,
            adjustment_requests,
            self.compartment_types_to_stratify,
        )
        self.stratify_entry_flows(
            stratification_name, strata_names, entry_proportions, requested_proportions
        )
        if self.death_flows.shape[0] > 0:
            self.stratify_death_flows(stratification_name, strata_names, adjustment_requests)
        self.stratify_universal_death_rate(
            stratification_name, strata_names, adjustment_requests, compartment_types_to_stratify
        )

        # if stratifying by strain
        self.strains = strata_names if stratification_name == "strain" else self.strains

        # check submitted mixing matrix and combine with existing matrix, if any
        self.prepare_mixing_matrix(mixing_matrix, stratification_name, strata_names)

        # prepare infectiousness levels attribute
        self.prepare_infectiousness_levels(
            stratification_name, strata_names, infectiousness_adjustments
        )

        # prepare strata equilibration target proportions
        if target_props:
            self.prepare_and_check_target_props(target_props, stratification_name, strata_names)

    """
    stratification checking methods
    """

    def prepare_and_check_stratification(
        self,
        _stratification_name,
        _strata_names,
        _compartment_types_to_stratify,
        _adjustment_requests,
        _target_props,
        _verbose,
    ):
        """
        initial preparation and checks of user-submitted arguments

        :param _stratification_name: str
            the name of the stratification - i.e. the reason for implementing this type of stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _compartment_types_to_stratify:
            see check_compartment_request
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        :param _verbose:
            see stratify
        :param _target_props:
            see stratify

        :return:
            _strata_names: list
                revised version of user request after adaptation to class requirements
            adjustment_requests:
                revised version of _adjustment_requests after adaptation to class requirements
        """

        # collate all the stratifications that have been implemented so far
        if not _compartment_types_to_stratify:
            self.full_stratification_list.append(_stratification_name)

        # report progress
        self.verbose = _verbose
        self.output_to_user(
            "\n___________________\nimplementing stratification for: %s" % _stratification_name
        )

        # deal with stratifications that have specific behaviour
        if _stratification_name == "age":
            _strata_names = self.check_age_stratification(
                _strata_names, _compartment_types_to_stratify
            )
        elif _stratification_name == "strain":
            self.output_to_user("implementing strain stratification with specific behaviour")

        # make sure the stratification name is a string
        if not isinstance(_stratification_name, str):
            _stratification_name = str(_stratification_name)
            self.output_to_user(
                "converting stratification name %s to string" % _stratification_name
            )

        # check target proportions correctly specified
        if _target_props:
            for restriction in _target_props:
                if not type(_target_props[restriction]) == dict:
                    raise TypeError("target proportions not provided as dictionary")
                elif type(_target_props[restriction]) == dict and any(
                    [
                        target_key not in _strata_names
                        for target_key in _target_props[restriction].keys()
                    ]
                ):
                    raise ValueError("requested target proportion strata not in requested strata")

        # ensure requested stratification hasn't previously been implemented
        if _stratification_name in self.all_stratifications.keys():
            raise ValueError(
                "requested stratification has already been implemented, please choose a different name"
            )

        # record stratification as model attribute, find the names to apply strata and check requests
        _strata_names = self.find_strata_names_from_input(_strata_names)
        self.all_stratifications[_stratification_name] = _strata_names
        _adjustment_requests = self.incorporate_alternative_overwrite_approach(_adjustment_requests)
        self.check_compartment_request(_compartment_types_to_stratify)
        self.check_parameter_adjustment_requests(_adjustment_requests, _strata_names)
        return _strata_names, _adjustment_requests

    def check_age_stratification(self, _strata_names, _compartment_types_to_stratify):
        """
        check that the user request meets the requirements for stratification by age

        :parameters: all parameters have come directly from the stratification (stratify) method unchanged and have been
            renamed with a preceding _ character
        :return: _strata_names: list
            revised names of the strata tiers to be implemented
        """
        self.output_to_user("implementing age stratification with specific behaviour")
        if len(_compartment_types_to_stratify) > 0:
            raise ValueError(
                "requested age stratification, but compartment request should be passed as empty vector "
                + "in order to apply to all compartments"
            )
        elif not all([isinstance(stratum, (int, float)) for stratum in _strata_names]):
            raise ValueError("inputs for age strata breakpoints are not numeric")
        if 0 not in _strata_names:
            _strata_names.append(0)
            self.output_to_user(
                "adding age stratum called '0' because not requested, which represents those aged "
                + "less than %s" % min(_strata_names)
            )
        if _strata_names != sorted(_strata_names):
            _strata_names = sorted(_strata_names)
            self.output_to_user(
                "requested age strata not ordered, so have been sorted to: %s" % _strata_names
            )
        return _strata_names

    def find_strata_names_from_input(self, _strata_names):
        """
        find the names of the strata to be implemented from a particular user request

        :parameters: list or alternative format to be adapted
            strata requested in the format provided by the user (except for age, which is dealth with in the preceding
            method)
        :return: strata_names: list
            modified list of strata to be implemented in model
        """
        if type(_strata_names) == int:
            _strata_names = numpy.arange(1, _strata_names + 1)
            self.output_to_user(
                "single integer provided as strata labels for stratification, hence strata "
                + "implemented will be integers from one to %s" % _strata_names
            )
        elif type(_strata_names) == float:
            raise ValueError(
                "single value passed as request for strata labels, but not an integer greater than "
                + "one, so unclear what to do - stratification failed"
            )
        elif type(_strata_names) == list and len(_strata_names) > 0:
            pass
        else:
            raise ValueError(
                "requested to stratify, but strata-level names not submitted in correct format"
            )
        for name in range(len(_strata_names)):
            _strata_names[name] = str(_strata_names[name])
            self.output_to_user("adding stratum: %s" % _strata_names[name])
        return _strata_names

    def incorporate_alternative_overwrite_approach(self, _adjustment_requests):
        """
        alternative approach to working out which parameters to overwrite
        can put a capital W at the string's end to indicate that it is an overwrite parameter, as an alternative to
        submitting a separate dictionary key to represent the strata which need to be overwritten

        :param _adjustment_requests: dict
            user-submitted version of adjustment requests
        :return: revised_adjustments: dict
            modified version of _adjustment_requests after working out whether any parameters began with W
        """

        # has to be constructed as a separate dictionary to avoid change of size during iteration
        revised_adjustments = {}
        for parameter in _adjustment_requests:
            revised_adjustments[parameter] = {}

            # ignore overwrite if submitted with the standard approach
            for stratum in _adjustment_requests[parameter]:
                if stratum == OVERWRITE_KEY:
                    continue

                # if the parameter ends in W, interpret as an overwrite parameter and added to this key
                elif stratum[-1] == OVERWRITE_CHARACTER:
                    if OVERWRITE_KEY not in revised_adjustments[parameter]:
                        revised_adjustments[parameter][OVERWRITE_KEY] = []
                    revised_adjustments[parameter][stratum[:-1]] = _adjustment_requests[parameter][
                        stratum
                    ]
                    revised_adjustments[parameter][OVERWRITE_KEY].append(stratum[:-1])

                # otherwise just accept the parameter in its submitted form
                else:
                    revised_adjustments[parameter][stratum] = _adjustment_requests[parameter][
                        stratum
                    ]
            if OVERWRITE_KEY not in revised_adjustments:
                revised_adjustments[OVERWRITE_KEY] = []
        return revised_adjustments

    def check_compartment_request(self, _compartment_types_to_stratify):
        """
        check the requested compartments to be stratified has been requested correctly

        :param _compartment_types_to_stratify: list
            the names of the compartment types that the requested stratification is intended to apply to
        """

        # if list of length zero passed, stratify all the compartment types in the model
        if len(_compartment_types_to_stratify) == 0:
            self.compartment_types_to_stratify = self.compartment_types
            self.output_to_user(
                "no compartment names specified for this stratification, "
                + "so stratification applied to all model compartments"
            )

        # otherwise check all the requested compartments are available and implement the user request
        elif any(
            [
                compartment not in self.compartment_types
                for compartment in self.compartment_types_to_stratify
            ]
        ):
            raise ValueError(
                "requested compartment or compartments to be stratified are not available in this model"
            )
        else:
            self.compartment_types_to_stratify = _compartment_types_to_stratify

    def check_parameter_adjustment_requests(self, _adjustment_requests, _strata_names):
        """
        check parameter adjustments have been requested appropriately and add parameter for any strata not referred to

        :param _adjustment_requests: dict
            version of the submitted adjustment_requests modified by incorporate_alternative_overwrite_approach
        :param _strata_names:
            see find_strata_names_from_input
        """
        for parameter in _adjustment_requests:
            if any(
                requested_stratum not in _strata_names + [OVERWRITE_KEY]
                for requested_stratum in _adjustment_requests[parameter]
            ):
                raise ValueError(
                    "a stratum was requested in adjustments that is not available in this stratification"
                )

    """
    stratification preparation methods
    """

    def set_ageing_rates(self, strata_names):
        """
        Set inter-compartmental flows for ageing from one stratum to the next.
        The ageing rate is proportional to the width of the age bracket.
        """
        ageing_flows = []
        for strata_idx in range(len(strata_names) - 1):
            start_age = int(strata_names[strata_idx])
            end_age = int(strata_names[strata_idx + 1])
            ageing_parameter_name = f"ageing{start_age}to{end_age}"
            ageing_rate = 1.0 / (end_age - start_age)
            self.parameters[ageing_parameter_name] = ageing_rate
            for compartment in self.compartment_names:
                ageing_flow = {
                    "type": Flow.STANDARD,
                    "parameter": ageing_parameter_name,
                    "origin": create_stratified_name(compartment, "age", start_age),
                    "to": create_stratified_name(compartment, "age", end_age),
                    "implement": len(self.all_stratifications),
                }
                ageing_flows.append(ageing_flow)

        self.transition_flows = self.transition_flows.append(ageing_flows)

    def prepare_starting_proportions(self, _strata_names, _requested_proportions):
        """
        prepare user inputs for starting proportions for the initial conditions to apply to the exact set of strata
            requested
        if one or more strata not specified, the proportion of the initial conditions allocated to that group will be
            the total unallocated population divided by the number of strata for which no request was specified

        :param _strata_names:
            see find_strata_names_from_input
        :param _requested_proportions: dict
            dictionary with keys for the stratum to assign starting population to and values the proportions to assign
        :return: dict
            revised dictionary of starting proportions after cleaning
        """
        self.output_to_user(
            "\n-----\ncalculating proportions of initial conditions to assign to each stratified starting compartment"
        )
        if any(stratum not in _strata_names for stratum in _requested_proportions):
            raise ValueError(
                "requested starting proportion for stratum that does not appear in requested strata"
            )
        if sum(_requested_proportions.values()) > 1.0:
            raise ValueError("requested starting proportions sum to a value greater than one")

        # assuming an equal proportion of the unallocated population if no request specified
        unrequested_strata = [
            stratum for stratum in _strata_names if stratum not in _requested_proportions
        ]
        unrequested_proportions = {}
        for stratum in unrequested_strata:
            starting_proportion = (1.0 - sum(_requested_proportions.values())) / len(
                unrequested_strata
            )
            unrequested_proportions[stratum] = starting_proportion
            self.output_to_user(
                "no starting proportion requested for %s stratum so provisionally allocated %s of total"
                % (stratum, round(starting_proportion, self.reporting_sigfigs))
            )

        # update specified proportions with inferred unspecified proportions
        _requested_proportions.update(unrequested_proportions)
        return _requested_proportions

    def stratify_compartments(
        self,
        stratification_name: str,
        strata_names: List[str],
        strata_proportions: Dict[str, float],
        compartments_to_stratify: List[str],
    ):
        """
        Stratify the model compartments into sub-compartments, based on the strata names provided,
        splitting the population according to the provided proprotions. Stratification will be applied
        to compartment_names and compartment_values.

        Only compartments specified in `self.compartment_types_to_stratify` will be stratified.
        """
        # Find the existing compartments that need stratification
        compartments_to_stratify = [
            c for c in self.compartment_names if find_stem(c) in compartments_to_stratify
        ]
        for compartment in compartments_to_stratify:
            # Add newm stratified compartment.
            for stratum in strata_names:
                name = create_stratified_name(compartment, stratification_name, stratum)
                idx = self.compartment_names.index(compartment)
                value = self.compartment_values[idx] * strata_proportions[stratum]
                self.add_compartment(name, value)

            # Remove the original compartment, since it has now been stratified.
            self.remove_compartment(compartment)

    def stratify_transition_flows(
        self,
        stratification_name: str,
        strata_names: List[str],
        adjustment_requests: Dict[str, Dict[str, float]],
        compartments_to_stratify: List[str],
    ):
        """
        Stratify flows depending on whether inflow, outflow or both need replication
        """
        flow_idxs = self.find_transition_indices_to_implement(back_one=1, include_change=True)
        all_new_flows = []
        for n_flow in flow_idxs:
            new_flows = []
            flow = self.transition_flows.iloc[n_flow]
            stratify_from = find_stem(flow.origin) in compartments_to_stratify
            stratify_to = find_stem(flow.to) in compartments_to_stratify
            if stratify_from or stratify_to:
                for stratum in strata_names:
                    # Find the flow's parameter name
                    parameter_name = self.add_adjusted_parameter(
                        flow.parameter, stratification_name, stratum, adjustment_requests,
                    )
                    if not parameter_name:
                        parameter_name = self.sort_absent_transition_parameter(
                            stratification_name,
                            strata_names,
                            stratum,
                            stratify_from,
                            stratify_to,
                            flow.parameter,
                        )

                    # Determine whether to and/or from compartments are stratified
                    from_compartment = (
                        create_stratified_name(flow.origin, stratification_name, stratum)
                        if stratify_from
                        else flow.origin
                    )
                    to_compartment = (
                        create_stratified_name(flow.to, stratification_name, stratum)
                        if stratify_to
                        else flow.to
                    )
                    # Add the new flow
                    strain = (
                        stratum
                        if stratification_name == "strain" and flow.type != Flow.STRATA_CHANGE
                        else flow.strain
                    )
                    new_flow = {
                        "type": flow.type,
                        "parameter": parameter_name,
                        "origin": from_compartment,
                        "to": to_compartment,
                        "implement": len(self.all_stratifications),
                        "strain": strain,
                    }
                    new_flows.append(new_flow)

            else:
                # If flow applies to a transition not involved in the stratification,
                # still increment to ensure that it is implemented.
                new_flow = flow.to_dict()
                new_flow["implement"] += 1
                new_flows.append(new_flow)

            # Update the customised flow functions.
            num_flows = len(self.transition_flows) + len(all_new_flows)
            for idx, new_flow in enumerate(new_flows):
                if new_flow["type"] == Flow.CUSTOM:
                    new_idx = num_flows + idx
                    self.customised_flow_functions[new_idx] = self.customised_flow_functions[n_flow]

            all_new_flows += new_flows

        if all_new_flows:
            self.transition_flows = self.transition_flows.append(all_new_flows, ignore_index=True)

    def add_adjusted_parameter(
        self, _unadjusted_parameter, _stratification_name, _stratum, _adjustment_requests
    ):
        """
        find the adjustment request that is relevant to a particular unadjusted parameter and stratum and add the
            parameter value (str for function or float) to the parameters dictionary attribute
        otherwise allow return of None

        :param _unadjusted_parameter:
            name of the unadjusted parameter value
        :param _stratification_name:
            see prepare_and_check_stratification
        :param _stratum:
            stratum being considered by the method calling this method
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        :return: parameter_adjustment_name: str or None
            if returned as None, assumption will be that the original, unstratified parameter should be used
            otherwise create a new parameter name and value and store away in the appropriate model structure
        """
        parameter_adjustment_name = None
        relevant_adjustment_request = self.find_relevant_adjustment_request(
            _adjustment_requests, _unadjusted_parameter
        )
        if relevant_adjustment_request is not None:
            parameter_adjustment_name = (
                create_stratified_name(_unadjusted_parameter, _stratification_name, _stratum)
                if _stratum in _adjustment_requests[relevant_adjustment_request]
                else _unadjusted_parameter
            )
            self.output_to_user(
                "\t parameter for %s stratum of %s stratification is called %s"
                % (_stratum, _stratification_name, parameter_adjustment_name)
            )
            if _stratum in _adjustment_requests[relevant_adjustment_request]:
                self.parameters[parameter_adjustment_name] = _adjustment_requests[
                    relevant_adjustment_request
                ][_stratum]

            # record the parameters that over-write the less stratified parameters closer to the trunk of the tree
            if (
                OVERWRITE_KEY in _adjustment_requests[relevant_adjustment_request]
                and _stratum in _adjustment_requests[relevant_adjustment_request][OVERWRITE_KEY]
            ):
                self.overwrite_parameters.append(parameter_adjustment_name)
        return parameter_adjustment_name

    def find_relevant_adjustment_request(self, _adjustment_requests, _unadjusted_parameter):
        """
        find the adjustment requests that are extensions of the base parameter type being considered
        expected behaviour is as follows:
        * if there are no submitted requests (keys to the adjustment requests) that are extensions of the unadjusted
            parameter, will return None
        * if there is one submitted request that is an extension of the unadjusted parameter, will return that parameter
        * if there are multiple submitted requests that are extensions to the unadjusted parameter and one is more
            stratified than any of the others (i.e. more instances of the "X" string), will return this most stratified
            parameter
        * if there are multiple submitted requests that are extensions to the unadjusted parameter and several of them
            are equal in having the greatest extent of stratification, will return the first one with the greatest
            length in the order of looping through the keys of the request dictionary

        :param _unadjusted_parameter:
            see add_adjusted_parameter
        :param _adjustment_requests:
            see prepare_and_check_stratification
        :return: str or None
            the key of the adjustment request that is applicable to the parameter of interest if any, otherwise None
        """

        # find all the requests that start with the parameter of interest and their level of stratification
        applicable_params = [
            param for param in _adjustment_requests if _unadjusted_parameter.startswith(param)
        ]
        applicable_param_lengths = [len(find_name_components(param)) for param in applicable_params]

        # find the first most stratified parameter
        return (
            applicable_params[applicable_param_lengths.index(max(applicable_param_lengths))]
            if applicable_param_lengths
            else None
        )

    def sort_absent_transition_parameter(
        self,
        _stratification_name,
        _strata_names,
        _stratum,
        _stratify_from,
        _stratify_to,
        unstratified_name,
    ):
        """
        work out what to do if a specific transition parameter adjustment has not been requested

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _stratum:
        :param _stratify_from:
            see add_stratified_flows
        :param _stratify_to:
            see add_stratified_flows
        :param unstratified_name: str
            the name of the parameter before the stratification is implemented
        :return: str
            parameter name for revised parameter than wasn't provided
        """

        # default behaviour if not specified is to split the parameter into equal parts if to compartment is split
        if not _stratify_from and _stratify_to:
            self.output_to_user(
                "\t splitting existing parameter value %s into %s equal parts"
                % (unstratified_name, len(_strata_names))
            )
            parameter_name = create_stratified_name(
                unstratified_name, _stratification_name, _stratum
            )
            self.parameters[parameter_name] = 1.0 / len(_strata_names)
            self.adaptation_functions[parameter_name] = create_multiplicative_function(
                1.0 / len(_strata_names)
            )
            return parameter_name

        # otherwise if no request, retain the existing parameter
        else:
            self.output_to_user("\tretaining existing parameter value %s" % unstratified_name)
            return unstratified_name

    def stratify_entry_flows(
        self, _stratification_name, _strata_names, _entry_proportions, _requested_proportions
    ):
        """
        stratify entry/recruitment/birth flows according to requested entry proportion adjustments
        again, may need to revise behaviour for what is done if some strata are requested but not others

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _entry_proportions: dict
            user requested proportions to enter to each stratum
        :param _requested_proportions:
            see prepare_starting_proportions
        :return:
            normalised dictionary of the compartments that the new entry flows should come in to
        """
        if self.entry_compartment in self.compartment_types_to_stratify:
            self.output_to_user(
                "\n-----\ncalculating proportions of births/recruitment to assign to each stratified entry compartment"
            )
            for stratum in _strata_names:
                entry_fraction_name = create_stratified_name(
                    "entry_fraction", _stratification_name, stratum
                )

                # specific behaviour for age stratification
                if _stratification_name == "age" and str(stratum) == "0":
                    self.parameters[entry_fraction_name] = 1.0
                    continue
                elif _stratification_name == "age":
                    self.parameters[entry_fraction_name] = 0.0
                    continue

                # where a request for splitting entry rates has been submitted
                elif stratum in _entry_proportions and type(_entry_proportions[stratum]) == float:
                    self.parameters[entry_fraction_name] = _entry_proportions[stratum]
                    self.output_to_user(
                        "assigning requested proportion %s of births/recruitment to %s stratum"
                        % (_entry_proportions[stratum], stratum)
                    )

                # if an incorrect string has been submitted by the user
                elif (
                    stratum in _entry_proportions
                    and type(_entry_proportions[stratum]) == str
                    and _entry_proportions[stratum] not in self.time_variants
                ):
                    raise ValueError(
                        "requested entry fraction function for %s stratum not available in time variants"
                    )

                # otherwise it must already be a defined function that can be called during integration
                elif stratum in _entry_proportions and type(_entry_proportions[stratum]) == str:
                    self.time_variants[entry_fraction_name] = self.time_variants[
                        _entry_proportions[stratum]
                    ]
                    self.output_to_user(
                        "function %s submitted for proportion of births assigned to %s"
                        % (_entry_proportions[stratum], stratum)
                    )
                    continue

                # otherwise if no request made
                else:
                    self.parameters[entry_fraction_name] = 1.0 / len(_strata_names)

    def stratify_death_flows(self, _stratification_name, _strata_names, _adjustment_requests):
        """
        add compartment-specific death flows to death_flows data frame attribute

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
             see find_strata_names_from_input
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        """
        for n_flow in self.find_death_indices_to_implement(back_one=1):

            # if the compartment with an additional death flow is being stratified
            if find_stem(self.death_flows.origin[n_flow]) in self.compartment_types_to_stratify:
                for stratum in _strata_names:

                    # get stratified parameter name if requested to stratify, otherwise use the unstratified one
                    parameter_name = self.add_adjusted_parameter(
                        self.death_flows.parameter[n_flow],
                        _stratification_name,
                        stratum,
                        _adjustment_requests,
                    )
                    if not parameter_name:
                        parameter_name = self.death_flows.parameter[n_flow]

                    # add the stratified flow to the death flows data frame
                    self.death_flows = self.death_flows.append(
                        {
                            "type": self.death_flows.type[n_flow],
                            "parameter": parameter_name,
                            "origin": create_stratified_name(
                                self.death_flows.origin[n_flow], _stratification_name, stratum
                            ),
                            "implement": len(self.all_stratifications),
                        },
                        ignore_index=True,
                    )

            # otherwise if not part of the stratification, accept the existing flow and increment the implement value
            else:
                new_flow = self.death_flows.loc[n_flow, :].to_dict()
                new_flow["implement"] += 1
                self.death_flows = self.death_flows.append(new_flow, ignore_index=True)

    def stratify_universal_death_rate(
        self,
        _stratification_name,
        _strata_names,
        _adjustment_requests,
        _compartment_types_to_stratify,
    ):
        """
        stratify the approach to universal, population-wide deaths (which can be made to vary by stratum)
        adjust every parameter that refers to the universal death rate, according to user request if submitted and
            otherwise populated with a value of one by default

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        :param _compartment_types_to_stratify:
            see above
        """
        if (
            _stratification_name not in self.full_stratification_list
            and "universal_death_rate" in _adjustment_requests
        ):
            raise ValueError(
                "universal death rate can only be stratified when applied to all compartment types"
            )
        elif _stratification_name not in self.full_stratification_list:
            self.output_to_user(
                "universal death rate not adjusted as stratification not applied to all compartments"
            )
            return

        # ensure baseline function available for modification in universal death rates
        self.adaptation_functions["universal_death_rateX"] = (
            self.time_variants["universal_death_rate"]
            if "universal_death_rate" in self.time_variants
            else lambda time: self.parameters["universal_death_rate"]
        )

        # if stratification applied to all compartment types
        for stratum in _strata_names:
            if (
                "universal_death_rate" in _adjustment_requests
                and stratum in _adjustment_requests["universal_death_rate"]
            ):
                stratum_name = create_stratum_name(_stratification_name, stratum, joining_string="")
                self.available_death_rates.append(stratum_name)

                # use existing function or create new one from constant as needed
                if type(_adjustment_requests["universal_death_rate"][stratum]) == str:
                    self.adaptation_functions[
                        "universal_death_rateX" + stratum_name
                    ] = self.time_variants[_adjustment_requests["universal_death_rate"][stratum]]
                elif isinstance(
                    _adjustment_requests["universal_death_rate"][stratum], (int, float)
                ):
                    self.adaptation_functions[
                        "universal_death_rateX" + stratum_name
                    ] = create_multiplicative_function(
                        self.time_variants[_adjustment_requests["universal_death_rate"][stratum]]
                    )

                # record the parameters that over-write the less stratified parameters closer to the trunk of the tree
                if (
                    OVERWRITE_KEY in _adjustment_requests["universal_death_rate"]
                    and stratum in _adjustment_requests["universal_death_rate"][OVERWRITE_KEY]
                ):
                    self.overwrite_parameters.append(
                        create_stratified_name(
                            "universal_death_rate", _stratification_name, stratum
                        )
                    )

    def prepare_mixing_matrix(self, _mixing_matrix, _stratification_name, _strata_names):
        """
        check that the mixing matrix has been correctly specified and call the other relevant functions

        :param _mixing_matrix: numpy array
            must be square
            represents the mixing of the strata within this stratification
        :param _stratification_name: str
            the name of the stratification - i.e. the reason for implementing this type of stratification
        :param _strata_names: list
            see find_strata_names_from_input
        """
        if _mixing_matrix is None:
            return
        elif type(_mixing_matrix) != numpy.ndarray:
            raise ValueError("submitted mixing matrix is wrong data type")
        elif len(_mixing_matrix.shape) != 2:
            raise ValueError("submitted mixing matrix is not two-dimensional")
        elif _mixing_matrix.shape[0] != _mixing_matrix.shape[1]:
            raise ValueError("submitted mixing is not square")
        elif _mixing_matrix.shape[0] != len(_strata_names):
            raise ValueError("mixing matrix does not sized to number of strata being implemented")
        self.combine_new_mixing_matrix_with_existing(
            _mixing_matrix, _stratification_name, _strata_names
        )

    def combine_new_mixing_matrix_with_existing(
        self, _mixing_matrix, _stratification_name, _strata_names
    ):
        """
        master mixing matrix function to take in a new mixing matrix and combine with the existing ones

        :param _mixing_matrix: numpy array
            array, which must be square representing the mixing of the strata within this stratification
        :param _stratification_name: str
            the name of the stratification - i.e. the reason for implementing this type of stratification
        :param _strata_names: list
            see find_strata_names_from_input
        """

        # if no mixing matrix yet, just convert the existing one to a dataframe
        if self.mixing_matrix is None:
            self.mixing_categories = [_stratification_name + "_" + i for i in _strata_names]
            self.mixing_matrix = _mixing_matrix

        # otherwise take the kronecker product to get the new mixing matrix
        else:
            self.mixing_categories = [
                old_strata + "X" + _stratification_name + "_" + new_strata
                for old_strata, new_strata in itertools.product(
                    self.mixing_categories, _strata_names
                )
            ]
            self.mixing_matrix = numpy.kron(self.mixing_matrix, _mixing_matrix)

    def prepare_infectiousness_levels(
        self, _stratification_name, _strata_names, _infectiousness_adjustments
    ):
        """
        store infectiousness adjustments as dictionary attribute to the model object, with first tier of keys the
            stratification and second tier the strata to be modified

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
             see find_strata_names_from_input
        :param _infectiousness_adjustments: dict
            requested adjustments to infectiousness for this stratification
        """
        if type(_infectiousness_adjustments) != dict:
            raise ValueError("infectiousness adjustments not submitted as dictionary")
        elif not all(key in _strata_names for key in _infectiousness_adjustments.keys()):
            raise ValueError("infectiousness adjustment key not in strata being implemented")
        else:
            for stratum in _infectiousness_adjustments:
                self.infectiousness_levels[
                    create_stratum_name(_stratification_name, stratum, joining_string="")
                ] = _infectiousness_adjustments[stratum]

    def prepare_and_check_target_props(self, _target_props, _stratification_name, _strata_names):
        """
        create the dictionary of dictionaries that contains the target values for equlibration

        :parameters:
            _target_props: dict
                user submitted dictionary with keys the restrictions by previously implemented strata that apply
            _stratification_name: str
                name of stratification process currently being implemented
            _strata_names: list
                list of the names of the strata being implemented under the current stratification process
        """
        self.target_props[_stratification_name] = {}
        for restriction in _target_props:
            self.target_props[_stratification_name][restriction] = {}

            # only need parameter values for the first n-1 strata, as the last one will be the remainder
            for stratum in _strata_names[:-1]:
                if stratum not in _target_props[restriction]:
                    raise ValueError(
                        "one or more of first n-1 strata being applied not in the target prop request"
                    )
                elif isinstance(_target_props[restriction][stratum], (float, int, str)):
                    self.target_props[_stratification_name][restriction][stratum] = _target_props[
                        restriction
                    ][stratum]
                else:
                    raise ValueError("target proportions specified with incorrect format for value")
                if (
                    type(_target_props[restriction][stratum]) == str
                    and _target_props[restriction][stratum] not in self.time_variants
                ):
                    raise ValueError("function for prevalence of %s not found" % stratum)
            if _strata_names[-1] in self.target_props:
                self.output_to_user(
                    "target proportion requested for stratum %s, but as last stratum"
                    % _strata_names[-1]
                    + " in request, this will be ignored and assigned the remainder to ensure sum to one"
                )

            # add the necessary flows to the transition data frame
            self.link_strata_with_flows(_stratification_name, _strata_names, restriction)

    def link_strata_with_flows(self, _stratification_name, _strata_names, _restriction):
        """
        add in sequential series of flows between neighbouring strata that transition people between the strata being
            implemented in this stratification stage

        :parameters:
            _stratification_name: str
                name of stratification currently being implemented
            _strata_names: list
                list of the strata being implemented in this stratification process
            _restriction: str
                name of previously implemented stratum that this equilibration flow applies to, if any, otherwise "all"
        """
        for compartment in self.unstratified_compartment_names:
            if _restriction in find_name_components(compartment) or _restriction == "all":
                for n_stratum in range(len(_strata_names[:-1])):
                    self.transition_flows = self.transition_flows.append(
                        {
                            "type": Flow.STRATA_CHANGE,
                            "parameter": _stratification_name
                            + "X"
                            + _restriction
                            + "X"
                            + _strata_names[n_stratum]
                            + "_"
                            + _strata_names[n_stratum + 1],
                            "origin": create_stratified_name(
                                compartment, _stratification_name, _strata_names[n_stratum]
                            ),
                            "to": create_stratified_name(
                                compartment, _stratification_name, _strata_names[n_stratum + 1]
                            ),
                            "implement": len(self.all_stratifications),
                            "strain": float("nan"),
                        },
                        ignore_index=True,
                    )

    """
    pre-integration methods
    """

    def prepare_to_run(self):
        """
        methods that can be run prior to integration to save various function calls being made at every time step
        """
        self.prepare_stratified_parameter_calculations()
        self.prepare_infectiousness_calculations()
        self.transition_indices_to_implement = self.find_transition_indices_to_implement()
        self.death_indices_to_implement = self.find_death_indices_to_implement()
        self.change_indices_to_implement = self.find_change_indices_to_implement()

        # ensure there is a universal death rate available even if the model hasn't been stratified at all
        if len(self.all_stratifications) == 0 and isinstance(
            self.parameters["universal_death_rate"], (float, int)
        ):
            self.final_parameter_functions["universal_death_rate"] = lambda time: self.parameters[
                "universal_death_rate"
            ]
        elif (
            len(self.all_stratifications) == 0
            and type(self.parameters["universal_death_rate"]) == str
        ):
            self.final_parameter_functions["universal_death_rate"] = self.adaptation_functions[
                "universal_death_rate"
            ]

        self.find_strata_indices()
        self.prepare_lookup_tables()

    def find_strata_indices(self):
        for stratif in self.all_stratifications:
            self.strata_indices[stratif] = {}
            for i_stratum, stratum in enumerate(self.all_stratifications[stratif]):
                self.strata_indices[stratif][stratum] = [
                    i_comp
                    for i_comp in range(len(self.compartment_names))
                    if create_stratum_name(
                        stratif, self.all_stratifications[stratif][i_stratum], joining_string=""
                    )
                    in find_name_components(self.compartment_names[i_comp])
                ]

    def prepare_stratified_parameter_calculations(self):
        """
        prior to integration commencing, work out what the components are of each parameter being implemented
        populates self.parameter_components even though it is not needed elsewhere, to allow that the components that
            were used to create each given parameter can be determined later
        """

        # create list of all the parameters that we need to find the set of adjustment functions for
        parameters_to_adjust = []

        transition_flow_indices = [
            n_flow
            for n_flow, flow in enumerate(self.transition_flows.type)
            if "change" not in flow
            and self.transition_flows.implement[n_flow] == len(self.all_stratifications)
        ]

        for n_flow in transition_flow_indices:
            if (
                self.transition_flows.implement[n_flow] == len(self.all_stratifications)
                and self.transition_flows.parameter[n_flow] not in parameters_to_adjust
            ):
                parameters_to_adjust.append(self.transition_flows.parameter[n_flow])
        for n_flow in range(self.death_flows.shape[0]):
            if (
                self.death_flows.implement[n_flow] == len(self.all_stratifications)
                and self.death_flows.parameter[n_flow] not in parameters_to_adjust
            ):
                parameters_to_adjust.append(self.death_flows.parameter[n_flow])

        # and adjust
        for parameter in parameters_to_adjust:
            self.parameter_components[parameter] = self.find_transition_components(parameter)
            self.create_transition_functions(parameter, self.parameter_components[parameter])

        # similarly for all model compartments
        for compartment in self.compartment_names:
            self.mortality_components[compartment] = self.find_mortality_components(compartment)
            if len(self.all_stratifications) > 0:
                self.create_mortality_functions(compartment, self.mortality_components[compartment])

    def find_mortality_components(self, _compartment):
        """
        find the sub-parameters for population-wide natural mortality that are relevant to a particular compartment
        used in prepare_stratified_parameter_calculations for creating functions to find the mortality rate for each
            compartment
        similar to find_transition_components, except being applied by compartment rather than parameter

        :param _compartment: str
            name of the compartment of interest
        :return: all_sub_parameters: list
            list of all the mortality-related sub-parameters for the compartment of interest
        """
        all_sub_parameters = []
        compartments_strata = find_name_components(_compartment)[1:]
        compartments_strata.reverse()
        compartments_strata.append("")

        # loop through each stratification of the parameter and adapt if the parameter is available
        for stratum in compartments_strata:
            if stratum in self.available_death_rates:
                all_sub_parameters.append("universal_death_rateX" + stratum)
            if "universal_death_rateX" + stratum in self.overwrite_parameters:
                break
        all_sub_parameters.reverse()
        return all_sub_parameters

    def create_mortality_functions(self, _compartment, _sub_parameters):
        """
        loop through all the components to the population-wide mortality and create the recursive functions

        :param _compartment: str
            name of the compartment of interest
        :param _sub_parameters: list
            the names of the functions that need to update the upstream parameters
        :return:
        """
        self.final_parameter_functions[
            "universal_death_rateX" + _compartment
        ] = self.adaptation_functions[_sub_parameters[0]]
        for component in _sub_parameters[1:]:

            # get the new function to act on the less stratified function (closer to the "tree-trunk")
            if component not in self.parameters:
                raise ValueError(
                    "parameter component %s not found in parameters attribute" % component
                )
            elif type(self.parameters[component]) == float:
                self.adaptation_functions[component] = create_multiplicative_function(
                    self.parameters[component]
                )
            elif type(self.parameters[component]) == str:
                self.adaptation_functions[component] = create_time_variant_multiplicative_function(
                    self.adaptation_functions[component]
                )
            else:
                raise ValueError("parameter component %s not appropriate format" % component)

            # create the composite function
            self.final_parameter_functions[
                "universal_death_rateX" + _compartment
            ] = create_function_of_function(
                self.adaptation_functions[component],
                self.final_parameter_functions["universal_death_rateX" + _compartment],
            )

    def find_transition_components(self, _parameter):
        """
        finds each of the strings for the functions acting on the next function in the sequence

        :param _parameter: str
            full name of the parameter of interest
        """
        sub_parameters = []

        # work backwards to allow stopping for overwriting requests, then reverse in preparation for function creation
        for x_instance in extract_reversed_x_positions(_parameter):
            component = _parameter[:x_instance]
            sub_parameters.append(component)
            if component in self.overwrite_parameters:
                break
        sub_parameters.reverse()
        return sub_parameters

    def create_transition_functions(self, _parameter, _sub_parameters):
        """
        builds up each parameter to be implemented as a function, recursively creating an outer function that calls the
            inner function

        :param _parameter: str
            full name of the parameter of interest
        :param _sub_parameters: list
            list of the strings representing the sub-parameters, including the base parameter as the stem and with all
                of the relevant strata in the stratification sequence following
        """

        # start from base value as a function of time, even if the time argument is ignored
        if isinstance(self.parameters[_sub_parameters[0]], (float, int)):
            self.final_parameter_functions[_parameter] = lambda time: self.parameters[
                _sub_parameters[0]
            ]
        elif type(self.parameters[_sub_parameters[0]]) == str:
            self.final_parameter_functions[_parameter] = self.adaptation_functions[
                _sub_parameters[0]
            ]

        # then cycle through other applicable components and extend function recursively, only if component available
        for component in _sub_parameters[1:]:

            # get the new function to act on the less stratified function (closer to the "tree-trunk")
            if component not in self.parameters:
                raise ValueError(
                    "parameter component %s not found in parameters attribute" % component
                )
            elif isinstance(self.parameters[component], float):
                self.adaptation_functions[component] = create_multiplicative_function(
                    self.parameters[component]
                )
            elif type(self.parameters[component]) == str:
                self.adaptation_functions[component] = create_time_variant_multiplicative_function(
                    self.time_variants[self.parameters[component]]
                )
            else:
                raise ValueError("parameter component %s not appropriate format" % component)

            # create the composite function
            self.final_parameter_functions[_parameter] = create_function_of_function(
                self.adaptation_functions[component], self.final_parameter_functions[_parameter]
            )

    def prepare_infectiousness_calculations(self):
        """
        master method to run all the code concerned with preparation for force of infection calculations
        """

        # infectiousness preparations
        self.prepare_all_infectiousness_multipliers()
        self.find_infectious_indices()

        # mixing preparations
        if self.mixing_matrix is not None:
            self.add_force_indices_to_transitions()
        self.find_mixing_denominators()

        # reconciling the strains and the mixing attributes together into one structure
        self.find_strain_mixing_multipliers()

    def prepare_all_infectiousness_multipliers(self):
        """
        find the infectiousness multipliers for each compartment being implemented in the model
        """

        # start from assumption that each compartment is fully and equally infectious
        self.infectiousness_multipliers = [1.0] * len(self.compartment_names)

        # if infectiousness modification requested for the compartment type, multiply through by the current value
        for n_comp, compartment in enumerate(self.compartment_names):
            for modifier in self.infectiousness_levels:
                if modifier in find_name_components(compartment):
                    self.infectiousness_multipliers[n_comp] *= self.infectiousness_levels[modifier]

        self.make_further_infectiousness_adjustments()

    def make_further_infectiousness_adjustments(self):
        """
        Work through specific requests for specific adjustments, to escape the requirement to only adjust compartment
        infectiousness according to stratification process - with all infectious compartments having the same
        adjustment.
        """
        for i_adjustment in range(len(self.individual_infectiousness_adjustments)):
            for i_comp, comp in enumerate(self.compartment_names):
                if all(
                    [
                        component in find_name_components(comp)
                        for component in self.individual_infectiousness_adjustments[i_adjustment][0]
                    ]
                ):
                    self.infectiousness_multipliers[
                        i_comp
                    ] = self.individual_infectiousness_adjustments[i_adjustment][1]

    def find_infectious_indices(self):
        """
        find the infectious indices by strain and overall, as opposed to just overall in EpiModel
        note that this changes the structure by one hierarchical level compared to EpiModel - in that previously we had
            self.infectious_indices a list of infectious indices and now it is has a dictionary structure at the highest
            level, followed by keys for each strain with values being lists that are equivalent to the
            self.infectious_indices list for the unstratified version
        """

        # find the indices for the compartments that are infectious across all strains
        self.infectious_indices["all_strains"] = self.find_all_infectious_indices()

        # then find the infectious compartment for each strain separately
        for strain in self.strains:
            self.infectious_indices[strain] = convert_boolean_list_to_indices(
                [
                    create_stratum_name("strain", strain, joining_string="")
                    in find_name_components(comp)
                    and i_comp in self.infectious_indices["all_strains"]
                    for i_comp, comp in enumerate(self.compartment_names)
                ]
            )

    def add_force_indices_to_transitions(self):
        """
        find the indices from the force of infection vector to be applied for each infection flow and populate to the
            force_index column of the flows frame
        """

        # identify the indices of all the infection-related flows to be implemented
        infection_flow_indices = [
            n_flow
            for n_flow, flow in enumerate(self.transition_flows.type)
            if "infection" in flow
            and self.transition_flows.implement[n_flow] == len(self.all_stratifications)
        ]

        # loop through and find the index of the mixing matrix applicable to the flow, of which there should be only one
        for n_flow in infection_flow_indices:
            found = False
            for i_group, force_group in enumerate(self.mixing_categories):
                if all(
                    stratum in find_name_components(self.transition_flows.origin[n_flow])
                    for stratum in find_name_components(force_group)
                ):
                    self.transition_flows.force_index[n_flow] = i_group
                    if found:
                        raise ValueError(
                            "mixing group found twice for transition flow number %s" % n_flow
                        )
                    found = True
                    continue
            if not found:
                raise ValueError("mixing group not found for transition flow number %s" % n_flow)

    def find_mixing_denominators(self):
        """
        for each mixing category, create a list of the compartment numbers that are relevant

        :return mixing_indices: list
            indices of the compartments that are applicable to a particular mixing category
        """
        if self.mixing_matrix is None:
            self.mixing_indices = {"all_population": range(len(self.compartment_names))}
        else:
            for category in self.mixing_categories:
                self.mixing_indices[category] = [
                    i_comp
                    for i_comp, compartment in enumerate(self.compartment_names)
                    if all(
                        [
                            component in find_name_components(compartment)
                            for component in find_name_components(category)
                        ]
                    )
                ]

    def find_strain_mixing_multipliers(self):
        """
        find the relevant indices to be used to calculate the force of infection contribution to each strain from each
            mixing category as a list of indices - and separately find multipliers as a list of the same length for
            their relative infectiousness extracted from self.infectiousness_multipliers
        """
        for strain in self.strains + ["all_strains"]:
            self.strain_mixing_elements[strain], self.strain_mixing_multipliers[strain] = {}, {}
            for category in (
                ["all_population"] if self.mixing_matrix is None else self.mixing_categories
            ):
                self.strain_mixing_elements[strain][category] = [
                    index
                    for index in self.mixing_indices[category]
                    if index in self.infectious_indices[strain]
                ]
                self.strain_mixing_multipliers[strain][category] = [
                    self.infectiousness_multipliers[i_comp]
                    for i_comp in self.strain_mixing_elements[strain][category]
                ]

    def find_transition_indices_to_implement(
        self, back_one: int = 0, include_change: bool = False
    ) -> List[int]:
        """
        Finds all the indices of the transition flows that need to be stratified,
        Overrides the version in the unstratified EpiModel

        :parameters:
            back_one: int
                number to subtract from self.all_stratification, which will be one if this method is being called after the
                    stratification has been added
            include_change: bool
                whether to include the strata_change transition flows
        :return: list
            list of indices of the flows that need to be stratified
        """
        return [
            idx
            for idx, flow in self.transition_flows.iterrows()
            if (flow.type != Flow.STRATA_CHANGE or include_change)
            and flow.implement == len(self.all_stratifications) - back_one
        ]

    def find_change_indices_to_implement(self, back_one=0):
        """
        find the indices of the equilibration flows to be applied in the transitions data frame

        :parameters:
            back_one: int
             see find_transition_indices_to_implement
        """
        return [
            idx
            for idx, flow in self.transition_flows.iterrows()
            if flow.type == Flow.STRATA_CHANGE
            and flow.implement == len(self.all_stratifications) - back_one
        ]

    def find_death_indices_to_implement(self, back_one=0):
        """
        find all the indices of the death flows that need to be stratified
        separated out as very short method in order that it can over-ride the version in the unstratified EpiModel

        :param back_one: int
            number to subtract from self.all_stratification, which will be one if this method is being called after the
                stratification has been added
        :return: list
            list of indices of the flows that need to be stratified
        """
        return self.death_flows[
            self.death_flows.implement == len(self.all_stratifications) - back_one
        ].index

    """
    methods to be called during the process of model running
    """

    # Cache return values to prevent wasteful re-computation - cache size is huge.
    # Floating point return type is 8 bytes, meaning 2**17 values is ~1MB of memory.
    @lru_cache(maxsize=2 ** 17)
    def get_parameter_value(self, _parameter, _time):
        """
        returns a parameter value by calling the function represented by its string within the parameter_functions
            attribute

        :param _parameter: str
            name of the parameter to be called (key to the parameter_functions dictionary)
        :param _time: float
            current time of model integration
        :return: float
            the parameter value needed
        """
        return self.final_parameter_functions[_parameter](_time)

    def find_infectious_population(self, _compartment_values):
        """
        find vectors for the total infectious populations and the total population that is needed in the case of
            frequency-dependent transmission

        :param _compartment_values: numpy array
            current values for the compartment sizes
        """
        mixing_categories = (
            ["all_population"] if self.mixing_matrix is None else self.mixing_categories
        )

        for strain in self.strains if self.strains else ["all_strains"]:
            self.infectious_populations[strain] = []
            for category in mixing_categories:
                self.infectious_populations[strain].append(
                    sum(
                        element_list_multiplication(
                            [
                                _compartment_values[i_comp]
                                for i_comp in self.strain_mixing_elements[strain][category]
                            ],
                            self.strain_mixing_multipliers[strain][category],
                        )
                    )
                )

        # Not sure which of these to use
        if type(_compartment_values) == list:
            _compartment_values = numpy.asarray(_compartment_values)
        self.infectious_denominator_shadow = \
            [sum(_compartment_values[self.mixing_indices[category]]) for category in self.mixing_indices]
        self.infectious_denominators = sum(_compartment_values)

    def find_infectious_multiplier(self, n_flow):
        """
        find the multiplier to account for the infectious population in dynamic flows

        :param n_flow: int
            index for the row of the transition_flows data frame
        :return:
            the total infectious quantity, whether that is the number or proportion of infectious persons
            needs to return as one for flows that are not transmission dynamic infectiousness flows
        """
        flow_type = self.transition_flows_dict["type"][n_flow]
        strain = self.transition_flows_dict["strain"][n_flow]
        force_index = self.transition_flows_dict["force_index"][n_flow]

        if "infection" not in flow_type:
            return 1.0
        strain = "all_strains" if not self.strains else strain
        mixing_elements = (
            [1.0] if self.mixing_matrix is None else self.mixing_matrix[force_index, :]
        )
        denominator = 1.0 if "_density" in flow_type else self.infectious_denominators

        return \
            sum(element_list_division(
                element_list_multiplication(
                    self.infectious_populations[strain],
                    mixing_elements
                ),
                self.infectious_denominator_shadow
            )
            )
        return sum(element_list_multiplication(self.infectious_populations[strain], mixing_elements)) \
               / denominator

    def prepare_time_step(self, _time):
        """
        Perform any tasks needed for execution of each integration time step
        """
        if self.dynamic_mixing_matrix:
            self.mixing_matrix = self.find_dynamic_mixing_matrix(_time)

    def find_dynamic_mixing_matrix(self, _time):
        """
        Function for overwriting in application to create time-variant mixing matrix
        """
        return self.mixing_matrix

    def get_compartment_death_rate(self, _compartment, _time):
        """
        find the universal or population-wide death rate for a particular compartment

        :param _compartment: str
            name of the compartment
        :param _time: float
            current integration time
        :return: float
            death rate
        """
        return (
            self.get_parameter_value("universal_death_rateX" + _compartment, _time)
            if len(self.all_stratifications) > 0
            else self.get_parameter_value("universal_death_rate", _time)
        )

    def apply_birth_rate(self, _ode_equations, _compartment_values, _time):
        """
        apply a population-wide death rate to all compartments
        all the entry_fraction proportions should be present in either parameters or time_variants given how they are
            created in the process of implementing stratification

        :parameters: all parameters have come directly from the apply_all_flow_types_to_odes method unchanged
        """

        # find the total number of births entering the system at the current time point
        total_births = self.find_total_births(_compartment_values, _time)

        # split the total births across entry compartments
        for compartment in [
            comp for comp in self.compartment_names if find_stem(comp) == self.entry_compartment
        ]:

            # calculate adjustment to original stem entry rate
            entry_fraction = 1.0
            for stratum in find_name_components(compartment)[1:]:
                entry_fraction *= self.get_single_parameter_component(
                    "entry_fractionX%s" % stratum, _time
                )

            # apply to that compartment
            _ode_equations = increment_list_by_index(
                _ode_equations,
                self.compartment_names.index(compartment),
                total_births * entry_fraction,
            )
        return _ode_equations

    def apply_change_rates(self, _ode_equations, _compartment_values, _time):
        """
        apply the transition rates that relate to equilibrating prevalence values for a particular stratification

        :parameters:
            _ode_equations: list
                working ode equations, to which transitions are being applied
            _compartment_values: list
                working compartment values
            _time: float
                current integration time value
        """

        # for each change flow being implemented
        for i_change in self.change_indices_to_implement:

            # split out the components of the transition string, which follow the standard 6-character string "change"
            stratification, restriction, transition = find_name_components(
                self.transition_flows.parameter[i_change]
            )
            origin_stratum, _ = transition.split("_")

            # find the distribution of the population across strata to be targeted
            _cumulative_target_props = self.find_target_strata_props(
                _time, restriction, stratification
            )

            # find the proportional distribution of the population across strata at the current time point
            _cumulative_strata_props = self.find_current_strata_props(
                _compartment_values, stratification, restriction
            )

            # work out which stratum and compartment transitions should be going from and to
            if _cumulative_strata_props[origin_stratum] > _cumulative_target_props[origin_stratum]:
                take_compartment, give_compartment, numerator, denominator = (
                    self.transition_flows.origin[i_change],
                    self.transition_flows.to[i_change],
                    _cumulative_strata_props[origin_stratum],
                    _cumulative_target_props[origin_stratum],
                )

            else:
                take_compartment, give_compartment, numerator, denominator = (
                    self.transition_flows.to[i_change],
                    self.transition_flows.origin[i_change],
                    1.0 - _cumulative_strata_props[origin_stratum],
                    1.0 - _cumulative_target_props[origin_stratum],
                )

            # calculate net flow
            net_flow = (
                numpy.log(numerator / denominator)
                / STRATA_EQUILIBRATION_FACTOR
                * _compartment_values[self.compartment_names.index(take_compartment)]
            )

            # update equations
            _ode_equations = increment_list_by_index(
                _ode_equations, self.compartment_names.index(take_compartment), -net_flow
            )
            _ode_equations = increment_list_by_index(
                _ode_equations, self.compartment_names.index(give_compartment), net_flow
            )
        return _ode_equations

    def find_target_strata_props(self, _time, _restriction, _stratification):
        """
        calculate the requested distribution of the population over the stratification that needs to be equilibrated
            over

        :parameters:
            _time: float
                current time value in integration
            _stratification: str
                name of the stratification over which the distribution of population is to be calculated
            _restriction: str
                name of the restriction stratification and the stratum joined with "_", if this is being applied
                if this is submitted as "all", the equilibration will be applied across all other strata
        """

        # for each applicable stratification, find target value for all strata, except the last one
        target_prop_values = {}
        for stratum in self.target_props[_stratification][_restriction]:
            target_prop_values[stratum] = (
                self.target_props[_stratification][_restriction][stratum]
                if type(self.target_props[_stratification][_restriction][stratum]) == float
                else self.time_variants[self.target_props[_stratification][_restriction][stratum]](
                    _time
                )
            )

        # check that prevalence values (including time-variant values) fall between zero and one
        if sum(target_prop_values.values()) > 1.0:
            raise ValueError(
                "total prevalence of first n-1 strata sums to more than one at time %s" % _time
            )
        elif any(target_prop_values.values()) < 0.0:
            raise ValueError("prevalence request of less than zero at time %s" % _time)

        # convert to dictionary of cumulative totals
        cumulative_target_props = create_cumulative_dict(target_prop_values)

        # add in a cumulative value of one for the last stratum
        cumulative_target_props.update({self.all_stratifications[_stratification][-1]: 1.0})
        return cumulative_target_props

    def find_current_strata_props(self, _compartment_values, _stratification, _restriction):
        """
        find the current distribution of the population across a particular stratification, which may or may not be
            restricted to a stratum of a previously implemented stratification process

        :parameters:
            _compartment_values: list
                current compartment values achieved during integration
            _stratification: str
                name of the stratification over which the distribution of population is to be calculated
            _restriction: str
                name of the restriction stratification and the stratum joined with "_", if this is being applied
                if this is submitted as "all", the equilibration will be applied across all other strata
        """

        # find the compartment indices applicable to the cross-stratification of interest (which may be all of them)
        if _restriction == "all":
            restriction_compartments = list(range(len(self.compartment_names)))
        else:
            restrict_stratification, restrict_stratum = _restriction.split("_")
            restriction_compartments = self.strata_indices[restrict_stratification][
                restrict_stratum
            ]

        # find current values of prevalence for the stratification for which prevalence values targeted
        current_strata_props = {}
        for stratum in self.all_stratifications[_stratification]:
            current_strata_props[stratum] = sum(
                [
                    _compartment_values[i_comp]
                    for i_comp in restriction_compartments
                    if i_comp in self.strata_indices[_stratification][stratum]
                ]
            ) / sum([_compartment_values[i_comp] for i_comp in restriction_compartments])

        return create_cumulative_dict(current_strata_props)


if __name__ == "__main__":

    def get_total_popsize(model, time):
        return sum(model.compartment_values)

    sir_model = StratifiedModel(
        numpy.linspace(0, 60 / 365, 61).tolist(),
        ["susceptible", "infectious", "recovered"],
        {"infectious": 0.001},
        {"beta": 400, "recovery": 365 / 13, "infect_death": 1},
        [
            {
                "type": "standard_flows",
                "parameter": "recovery",
                "origin": "infectious",
                "to": "recovered",
            },
            {
                "type": "infection_density",
                "parameter": "beta",
                "origin": "susceptible",
                "to": "infectious",
            },
            {"type": "compartment_death", "parameter": "infect_death", "origin": "infectious"},
        ],
        output_connections={
            "incidence": {"origin": "susceptible", "to": "infectious"},
            "incidence_hiv_positive": {
                "origin": "susceptible",
                "to": "infectious",
                "origin_condition": "hiv_positive",
                "to_condition": "hiv_positive",
            },
        },
        verbose=False,
        derived_output_functions={"population": get_total_popsize},
        death_output_categories=((), ("hiv_positive",)),
    )
    # sir_model.adaptation_functions["increment_by_one"] = create_additive_function(1.)

    # hiv_mixing = numpy.ones(4).reshape(2, 2)
    hiv_mixing = None

    temp_function = lambda time: 0.4
    sir_model.time_variants["temp_function"] = temp_function

    age_mixing = None
    sir_model.stratify(
        "age",
        [5, 10],
        [],
        {},
        {"recovery": {"5": 0.5, "10": 0.8}},
        infectiousness_adjustments={"5": 0.8},
        mixing_matrix=age_mixing,
        verbose=False,
    )

    sir_model.stratify(
        "hiv",
        ["negative", "positive"],
        [],
        {"negative": 0.6},
        {
            "recovery": {"negative": "increment_by_one", "positive": 0.5},
            "infect_death": {"negative": 0.5},
            "entry_fraction": {"negative": 0.6, "positive": 0.4},
        },
        adjustment_requests={"recovery": {"negative": 0.7}},
        infectiousness_adjustments={"positive": 0.5},
        mixing_matrix=hiv_mixing,
        verbose=False,
        target_props={"all": {"negative": 0.5}},
    )

    # sir_model.stratify("strain", ["sensitive", "resistant"], ["infectious"],
    #                    adjustment_requests={"recoveryXhiv_negative": {"sensitive": 0.9},
    #                                         "recovery": {"sensitive": 0.8}},
    #                    requested_proportions={}, verbose=False)

    sir_model.transition_flows.to_csv("transitions.csv")

    sir_model.run_model()

    # create_flowchart(sir_model, name="sir_model_diagram")

    # create_flowchart(sir_model)
    #
    sir_model.plot_compartment_size(["infectious", "hiv_positive"])
