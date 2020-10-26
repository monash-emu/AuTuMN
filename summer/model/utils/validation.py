"""
Functions to validate the inputs to a model.
Validation performed using Cerberus: https://docs.python-cerberus.org/en/stable/index.html
"""
from typing import List

from cerberus import Validator
import numpy as np


from summer.constants import (
    Compartment,
    Flow,
    BirthApproach,
    IntegrationType,
)


def validate_stratify(
    model,
    stratification_name,
    strata_request,
    compartments_to_stratify,
    requested_proportions,
    adjustment_requests,
    infectiousness_adjustments,
    mixing_matrix,
):
    schema = get_stratify_schema(
        model, stratification_name, strata_request, compartments_to_stratify
    )
    validator = Validator(schema, allow_unknown=True, require_all=True)
    stratify_data = {
        "stratification_name": stratification_name,
        "strata_request": strata_request,
        "compartments_to_stratify": compartments_to_stratify,
        "requested_proportions": requested_proportions,
        "adjustment_requests": adjustment_requests,
        "infectiousness_adjustments": infectiousness_adjustments,
        "mixing_matrix": mixing_matrix,
    }
    is_valid = validator.validate(stratify_data)
    if not is_valid:
        errors = validator.errors
        raise ValidationException(errors)


def get_stratify_schema(model, stratification_name, strata_names, compartments_to_stratify):
    """
    Schema used to validate model attributes during initialization.
    """
    prev_strat_names = [s.name for s in model.stratifications]
    strata_names_strs = [str(s) for s in strata_names]
    is_full_stratified = compartments_to_stratify == model.original_compartment_names
    return {
        "stratification_name": {
            "type": "string",
            "forbidden": prev_strat_names,
        },
        "strata_request": {
            "type": "list",
            "check_with": check_strata_request(
                stratification_name, compartments_to_stratify, model.original_compartment_names
            ),
            "schema": {"type": ["string", "integer"]},
        },
        "compartments_to_stratify": {
            "type": "list",
            "schema": {"type": "string"},
            "allowed": model.original_compartment_names,
        },
        "requested_proportions": {
            "type": "dict",
            "valuesrules": {"type": ["integer", "float"]},
            "keysrules": {"allowed": strata_names_strs},
        },
        "adjustment_requests": {
            "type": "dict",
            "keysrules": {
                # FIXME: Not quite right... yet.
                # "allowed": list(model.parameters.keys()),
                "check_with": check_adj_request_key(compartments_to_stratify),
            },
            "valuesrules": {
                "type": "dict",
                "keysrules": {"allowed": strata_names_strs + [f"{s}W" for s in strata_names_strs]},
                "valuesrules": {
                    "type": ["integer", "float", "string"],
                    "check_with": check_time_variant_key(model.time_variants),
                },
            },
        },
        "infectiousness_adjustments": {
            "type": "dict",
            "valuesrules": {"type": ["integer", "float"]},
            "keysrules": {"allowed": strata_names_strs},
        },
        "mixing_matrix": {
            "nullable": True,
            "check_with": check_mixing_matrix(strata_names, is_full_stratified),
        },
    }


def check_adj_request_key(compartments_to_stratify):
    """
    Check adjustment request keys
    """

    def _check(field, value, error):
        if not compartments_to_stratify and value == "universal_death_rate":
            error(
                field,
                "Universal death rate can only be adjusted for when all compartments are being stratified",
            )

    return _check


def check_strata_request(strat_name, compartments_to_stratify, model_compartments):
    """
    Strata requested must be well formed.
    """

    def _check(field, value, error):
        if strat_name == "age":
            if not min([int(s) for s in value]) == 0:
                error(field, "First age strata must be '0'")
            if compartments_to_stratify and not compartments_to_stratify == model_compartments:
                error(field, "Age stratification must be applied to all compartments")

    return _check


def check_time_variant_key(time_variants: dict):
    """
    Ensure value is a key in time variants if it is a string
    """

    def _check(field, value, error):
        if type(value) is str and value not in time_variants:
            error(field, "String value must be found in time variants dict.")

    return _check


def check_mixing_matrix(strata_names: List[str], is_full_stratified: bool):
    """
    Ensure mixing matrix is correctly specified
    """
    num_strata = len(strata_names)

    def _check(field, value, error):
        if value is None:
            return  # This is fine

        if not is_full_stratified:
            error(field, "Mixing matrix can only be applied to full stratifications.")

        if callable(value):
            # Dynamic mixing matrix
            mm = value(0)
        else:
            # Static mixing matrix
            mm = value

        if not type(mm) is np.ndarray:
            error(field, "Mixing matrix must be Numpy array (or None)")
        elif mm.shape != (num_strata, num_strata):
            error(field, f"Mixing matrix must have shape ({num_strata}, {num_strata})")

    return _check


def validate_model(model_kwargs):
    """
    Throws an error if the model's initial data is invalid.
    """
    starting_population = model_kwargs.get("starting_population")
    parameters = model_kwargs.get("parameters", {})
    compartment_names = model_kwargs.get("compartment_names", [])
    parameter_names = [*parameters.keys()]
    schema = get_model_schema(starting_population, parameter_names, compartment_names)
    validator = Validator(schema, allow_unknown=True, require_all=True)
    is_valid = validator.validate(model_kwargs)
    if not is_valid:
        errors = validator.errors
        raise ValidationException(errors)

    # Validate times seperately because Cerberus is slow.
    times = model_kwargs["times"]
    assert type(times) is np.ndarray, "Times must be a NumPy array."
    assert times.dtype == np.dtype("float64"), "Times must be float64."


def get_model_schema(starting_population, parameter_names, compartment_names):
    """
    Schema used to validate model attributes during initialization.
    """
    return {
        "starting_population": {"type": "integer"},
        "entry_compartment": {"type": "string", "allowed": compartment_names},
        "birth_approach": {
            "type": "string",
            "allowed": [
                BirthApproach.ADD_CRUDE,
                BirthApproach.REPLACE_DEATHS,
                BirthApproach.NO_BIRTH,
            ],
        },
        "compartment_names": {"type": "list", "schema": {"type": "string"}},
        "infectious_compartments": {
            "type": "list",
            "schema": {"type": "string"},
            "allowed": compartment_names,
        },
        "initial_conditions": {
            "type": "dict",
            "valuesrules": {"anyof_type": ["integer", "float"]},
            "check_with": check_initial_conditions(starting_population, compartment_names),
        },
        "requested_flows": {
            "type": "list",
            "check_with": check_requested_flows(parameter_names, compartment_names),
            "schema": {
                "type": "dict",
                "schema": {
                    "type": {
                        "type": "string",
                        "allowed": [
                            Flow.STANDARD,
                            Flow.INFECTION_FREQUENCY,
                            Flow.INFECTION_DENSITY,
                            Flow.DEATH,
                            Flow.IMPORT,
                        ],
                    },
                    "parameter": {"type": "string"},
                    "origin": {"type": "string", "required": False},
                    "to": {"type": "string", "required": False},
                },
            },
        },
    }


def check_requested_flows(parameter_names, compartment_names):
    """
    Validate flows
    """

    def _check(field, value, error):
        # Validate flows
        for flow in value:
            if "origin" in flow and flow["origin"] not in compartment_names:
                error(field, "From compartment name not found in compartment types")
            if "to" in flow and flow["to"] not in compartment_names:
                error(field, "To compartment name not found in compartment types")

    return _check


def check_initial_conditions(starting_population, compartment_names):
    """
    Ensure initial conditions are well formed, and do not exceed population numbers.
    """

    def _check(field, value, error):
        try:
            is_pop_too_small = sum(value.values()) > starting_population
            if is_pop_too_small:
                error(
                    field,
                    "Initial condition population exceeds total starting population.",
                )

            if not all([c in compartment_names for c in value.keys()]):
                error(
                    field,
                    "Initial condition compartment name is not one of the listed compartment types",
                )
        except TypeError:
            error(field, "Could not check initial conditions.")

    return _check


def check_times(field, value, error):
    """
    Ensure times are sorted in ascending order.
    """
    if sorted(value) != value:
        error(field, "Integration times are not in order")


class ValidationException(Exception):
    """
    Raised when user-defined data is found to be invalid.
    """

    pass
