from inspect import getfullargspec
from pydantic import BaseModel
from typing import Union

from summer import CompartmentalModel
from summer.parameters import Parameter, Function
from computegraph import ComputeGraph
from computegraph.utils import expand_nested_dict, is_var


class ModelBuilder:
    def __init__(self, model: CompartmentalModel, params: dict, pyd_params: BaseModel):
        self.model = model
        self.params = params
        self.params_expanded = expand_nested_dict(params, include_parents=True)
        self.pyd_params = pyd_params
        self.input_graph = {}

        self.required_outputs = set()

    def add_output(self, key, graph_obj):
        if key in self.input_graph:
            raise Exception(f"Key {key} already exists in graph as {self.input_graph[key]}")
        self.input_graph[key] = graph_obj

    def finalize(self):
        self.run = get_full_runner(self)

    def _get_func_args(self, key):
        return [Parameter(k) for k in [*self.params_expanded[key]]]

    def _get_func_kwargs(self, key):
        return {k: Parameter(k) for k in [*self.params_expanded[key]]}

    def _find_key_from_obj(self, obj):
        return find_key_from_obj(obj, self.pyd_params, self.params, None)

    def param(self, key):
        """Get a Parameter (computegraph Variable) for the given key
        If this key is not contained in the initial parameters, register it
        as a required additional parameter

        Args:
            key: Key of the parameter

        Returns:
            computegraph Parameter
        """
        if is_var(key, "parameters"):
            key = key.name
        if key not in self.params_expanded:
            if key not in self.required_outputs:
                self.required_outputs.add(key)
        return Parameter(key)

    def get_output(self, key):
        if key not in self.input_graph:
            raise KeyError(f"{key} does not exist in builder outputs")
        return Parameter(key)

    def _get_value(self, key: str):
        """Return the initial parameter value for the given key

        Args:
            key (str): Parameter key
        """
        if key in self.params_expanded:
            return self.params_expanded[key]
        else:
            raise KeyError("Key not found in initial parameters", key)

    def get_mapped_func(self, func: callable, param_obj: BaseModel, kwargs=None):
        argspec = getfullargspec(func)

        kwargs = {} if kwargs is None else kwargs

        supplied_argkeys = list(kwargs)

        msg = f"Function arguments {argspec} do not match pydantic object {param_obj}"
        assert all(
            [hasattr(param_obj, arg) for arg in argspec.args if arg not in supplied_argkeys]
        ), msg
        base_key = self._find_key_from_obj(param_obj)
        mapped_args = {arg: Parameter(f"{base_key}.{arg}") for arg in argspec.args}
        mapped_args.update(kwargs)
        return Function(func, [], mapped_args)


def find_key_from_obj(obj, pydparams, params, layer=None, is_dict=False):
    if layer is None:
        layer = []
    for k, v in params.items():
        if is_dict:
            cur_pydobj = pydparams[k]
        else:
            cur_pydobj = getattr(pydparams, k)
        if cur_pydobj is obj:
            if isinstance(cur_pydobj, BaseModel):
                return ".".join(layer + [k])
            else:
                raise Exception("Try using subkeys")
        if isinstance(cur_pydobj, dict):
            res = find_key_from_obj(obj, cur_pydobj, v, layer + [k], True)
            if res is not None:
                return res
        elif isinstance(cur_pydobj, BaseModel):
            assert isinstance(v, dict)
            res = find_key_from_obj(obj, cur_pydobj, v, layer + [k])
            if res is not None:
                return res
    if layer is None:
        raise Exception(f"Unable to match {obj} in parameters dictionary")


def get_full_runner(builder, use_jax=False):
    graph_run = ComputeGraph(builder.input_graph).get_callable()

    if use_jax:
        from summer.runner.jax.util import get_runner
        from summer.runner.jax.model_impl import build_run_model

        jrunner = get_runner(builder.model)
        jax_run_func, jax_runner_dict = build_run_model(jrunner)

    model_input_p = [p.name for p in list(builder.model.get_input_parameters())]

    def run_everything(param_updates=None, **kwargs):

        parameters = builder.params_expanded.copy()
        if param_updates is not None:
            parameters.update(param_updates)

        graph_outputs = graph_run(parameters=parameters)
        parameters.update(graph_outputs)

        model_params = {k: v for k, v in parameters.items() if k in model_input_p}

        if use_jax:
            return jax_run_func(parameters=model_params)
        else:
            builder.model.run(parameters=model_params, **kwargs)
            return builder.model

    def run_inputs(param_updates=None):
        parameters = builder.params_expanded.copy()
        if param_updates is not None:
            parameters.update(param_updates)

        graph_outputs = graph_run(parameters=parameters)
        parameters.update(graph_outputs)

        model_params = {k: v for k, v in parameters.items() if k in model_input_p}
        return model_params

    if use_jax:
        return run_everything, run_inputs, jax_run_func, jax_runner_dict
    else:
        return run_everything


# This union type should be checked against for anything expects
# to able to call model.run() and have a CompartmentalModel returned
RunnableModel = Union[CompartmentalModel, ModelBuilder]
