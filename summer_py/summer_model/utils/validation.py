"""
Functions to validate the inputs to a model.
Validation performed using Cerberus: https://docs.python-cerberus.org/en/stable/index.html
"""
from cerberus import Validator

from summer_py.constants import Compartment, Flow, BirthApproach, Stratification, IntegrationType


def validate_model(model):
    """
    Throws an error if the model's initial data is invalid.
    """
    schema = get_model_schema(model)
    validator = Validator(schema, allow_unknown=True, require_all=True)
    model_data = model.__dict__
    is_valid = validator.validate(model_data)
    if not is_valid:
        errors = validator.errors
        raise ValidationException(errors)


def get_model_schema(model):
    """
    Schema used to validate model attributes during initialization.
    """
    return {
        "verbose": {"type": "boolean"},
        "ticker": {"type": "boolean"},
        "reporting_sigfigs": {"type": "integer"},
        "starting_population": {"type": "integer"},
        "entry_compartment": {"type": "string"},
        "birth_approach": {
            "type": "string",
            "allowed": [
                BirthApproach.ADD_CRUDE,
                BirthApproach.REPLACE_DEATHS,
                BirthApproach.NO_BIRTH,
            ],
        },
        "times": {
            "type": "list",
            "schema": {"anyof_type": ["integer", "float"]},
            "check_with": check_times,
        },
        "compartment_types": {"type": "list", "schema": {"type": "string"}},
        "infectious_compartment": {
            "type": "list",
            "schema": {"type": "string"},
            "allowed": model.compartment_types,
        },
        "initial_conditions": {
            "type": "dict",
            "valueschema": {"anyof_type": ["integer", "float"]},
            "check_with": check_initial_conditions(model),
        },
        "requested_flows": {
            "type": "list",
            "check_with": check_flows(model),
            "schema": {
                "type": "dict",
                "schema": {
                    "type": {
                        "type": "string",
                        "allowed": [
                            Flow.CUSTOM,
                            Flow.STANDARD,
                            Flow.INFECTION_FREQUENCY,
                            Flow.INFECTION_DENSITY,
                            Flow.COMPARTMENT_DEATH,
                        ],
                    },
                    "parameter": {"type": "string"},
                    "origin": {"type": "string"},
                    "to": {"type": "string", "required": False},
                },
            },
        },
        "output_connections": {
            "type": "dict",
            "valueschema": {
                "type": "dict",
                "schema": {
                    "origin": {"type": "string"},
                    "to": {"type": "string"},
                    "origin_condition": {"type": "string", "required": False},
                    "to_condition": {"type": "string", "required": False},
                },
            },
        },
        "death_output_categories": {
            "type": "list",
            "schema": {"type": "list", "schema": {"type": "string"}},
        },
        "derived_output_functions": {"type": "dict", "check_with": check_derived_output_functions},
    }


def check_flows(model):
    """
    Validate flows
    """

    def _check(field, value, error):
        # Validate flows
        for flow in value:
            is_missing_params = (
                flow["parameter"] not in model.parameters
                and flow["parameter"] not in model.time_variants
            )
            if is_missing_params:
                error(field, "Flow parameter not found in parameter list")
            if flow["origin"] not in model.compartment_types:
                error(field, "From compartment name not found in compartment types")
            if "to" in flow and flow["to"] not in model.compartment_types:
                error(field, "To compartment name not found in compartment types")

            # Customized flows must have functions
            if flow["type"] == Flow.CUSTOM:
                if "function" not in flow.keys():
                    error(
                        field,
                        "A customised flow requires a function to be specified in user request dictionary.",
                    )
                elif not callable(flow["function"]):
                    error(field, "value of 'function' key must be a function")

    return _check


def check_initial_conditions(model):
    """
    Ensure initial conditions are well formed, and do not exceed population numbers.
    """

    def _check(field, value, error):
        try:
            is_pop_too_small = sum(value.values()) > model.starting_population
            if is_pop_too_small:
                error(field, "Initial condition population exceeds total starting population.")

            if not all([c in model.compartment_types for c in value.keys()]):
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


def check_derived_output_functions(field, value, error):
    """
    Ensure every item in dict is a function.
    """
    if not all([callable(f) for f in value.values()]):
        error(field, "Must be a dict of functions.")


class ValidationException(Exception):
    """
    Raised when user-defined data is found to be invalid.
    """

    pass
