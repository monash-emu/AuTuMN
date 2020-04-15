"""
Cerberus schema building utilities
"""
PRIMITIVES = [int, float, bool, str]


def build_schema(**schema):
    return _build_schema(schema)


def _build_schema(schema):
    if schema in PRIMITIVES:
        return build_primitive(schema)

    s_type = type(schema)
    if s_type in [Dict, List, DictGeneric]:
        return schema.build_schema()

    assert type(schema) is dict, "Schema must be a dict"
    cerberus_schema = {}
    for k, v in schema.items():
        if v in PRIMITIVES:
            cerberus_schema[k] = build_primitive(v)
        else:
            v_type = type(v)
            if v_type not in [Dict, List, DictGeneric]:
                raise ValueError(f"Could not build schema from type {k}: {v_type}")

            cerberus_schema[k] = v.build_schema()

    return cerberus_schema


def build_primitive(v_type):
    if v_type is int:
        return {"type": "integer"}
    elif v_type is float:
        return {"type": "float"}
    elif v_type is bool:
        return {"type": "boolean"}
    elif v_type is str:
        return {"type": "string"}
    else:
        raise ValueError(f"Could not find type {v_type}")


class List:
    def __init__(self, arg_schema):
        self.arg_schema = arg_schema

    def build_schema(self):
        return {"type": "list", "schema": _build_schema(self.arg_schema)}


class Dict:
    def __init__(self, **kwargs):
        self.schema = kwargs

    def build_schema(self):
        return {"type": "dict", "schema": _build_schema(self.schema)}


class DictGeneric:
    def __init__(self, key_schema, value_schema):
        self.key_schema = key_schema
        self.value_schema = value_schema

    def build_schema(self):
        return {
            "type": "dict",
            "valuesrules": _build_schema(self.value_schema),
            "keysrules": _build_schema(self.key_schema),
        }
