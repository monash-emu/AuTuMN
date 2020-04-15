from autumn.tool_kit import schema_builder as sb
from typing import List, Dict


def test_schema_builder():
    cerberus_schema = sb.build_schema(**INPUT_SCHEMA)
    assert cerberus_schema == EXPECTED_SCHEMA


INPUT_SCHEMA = {
    "translations": sb.DictGeneric(str, str),
    "outputs_to_plot": sb.List(sb.Dict(name=str)),
    "pop_distribution_strata": sb.List(str),
    "prevalence_combos": sb.List(sb.List(str)),
    "input_function": sb.Dict(start_time=float, func_names=sb.List(str)),
    "parameter_category_values": sb.Dict(time=float, param_names=sb.List(str)),
}


EXPECTED_SCHEMA = {
    "translations": {
        "type": "dict",
        "valuesrules": {"type": "string"},
        "keysrules": {"type": "string"},
    },
    "outputs_to_plot": {
        "type": "list",
        "schema": {"type": "dict", "schema": {"name": {"type": "string"},},},
    },
    "pop_distribution_strata": {"type": "list", "schema": {"type": "string"}},
    "prevalence_combos": {
        "type": "list",
        "schema": {"type": "list", "schema": {"type": "string"}},
    },
    "input_function": {
        "type": "dict",
        "schema": {
            "start_time": {"type": "float"},
            "func_names": {"type": "list", "schema": {"type": "string"}},
        },
    },
    "parameter_category_values": {
        "type": "dict",
        "schema": {
            "time": {"type": "float"},
            "param_names": {"type": "list", "schema": {"type": "string"}},
        },
    },
}
