import numpy
from scipy.integrate import odeint, solve_ivp, quad
import matplotlib.pyplot
import copy
import pandas as pd
from graphviz import Digraph
from sqlalchemy import create_engine
import os
from sqlalchemy import FLOAT

# set path - sachin
os.environ["PATH"] += os.pathsep + 'C:/Users/swas0001/graphviz-2.38/release/bin'

# set path - james desktop
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

# set path - romain desktop
os.environ["PATH"] += os.pathsep + 'C:/Users/rrag0004/Models/graphviz-2.38/release/bin'


def find_stem(stratified_string):
    """
    find the stem of the compartment name as the text leading up to the first occurrence of "X"
    """
    first_x_location = stratified_string.find("X")
    return stratified_string if first_x_location == -1 else stratified_string[: first_x_location]


def create_stratified_name(stem, stratification_name, stratum_name):
    """
    function to generate a standardised stratified compartment name
    """
    return stem + create_stratum_name(stratification_name, stratum_name)


def create_stratum_name(stratification_name, stratum_name):
    """
    generate the name just for the particular stratification
    """
    return "X%s_%s" % (stratification_name, str(stratum_name))


def extract_x_positions(parameter):
    """
    find the positions within a string which are X and return as list reversed, including length of list
    """
    result = [loc for loc in range(len(parameter)) if parameter[loc] == "X"]
    result.append(len(parameter))
    return result


def extract_reversed_x_positions(parameter):
    """
    find the positions within a string which are X and return as list reversed, including length of list
    """
    result = extract_x_positions(parameter)
    result.reverse()
    return result


def increment_compartment(ode_equations, compartment_number, increment):
    """
    general method to increment the odes by a value specified as an argument
    """
    ode_equations[compartment_number] += increment
    return ode_equations


def normalise_dict(value_dict):
    """
    simple function to normalise the values from a list
    """
    return {key: value_dict[key] / sum(value_dict.values()) for key in value_dict}


def num_str(f):
    """
    currently unused function to convert number to a readable string
    """
    abs_f = abs(f)
    if abs_f > 1e9:
        return "%.1fB" % (f / 1e9)
    if abs_f > 1e6:
        return "%.1fM" % (f / 1e6)
    if abs_f > 1e3:
        return "%.1fK" % (f / 1e3)
    if abs_f > 100.:
        return "%.0f" % f
    if abs_f > 0.5:
        return "%.1f" % f
    if abs_f > 5e-2:
        return "%.2f" % f
    if abs_f > 5e-4:
        return "%.4f" % f
    if abs_f > 5e-6:
        return "%.6f" % f
    return str(f)


def add_w_to_param_names(parameter_dict):
    """
    add a W string to the end of the parameter name to indicate that we should over-write up the chain
    """
    return {str(age_group) + "W": value for age_group, value in parameter_dict.items()}


def change_parameter_unit(parameter_dict, unit_change=365.25):
    """
    adapt the latency parameters from the earlier functions according to whether they are needed as by year rather than
    by day
    """
    return {param_key: param_value * unit_change for param_key, param_value in parameter_dict.items()}


def create_step_function_from_dict(input_dict):
    """
    create a step function out of dictionary with numeric keys and values, where the keys determine the values of the
    independent variable at which the steps between the output values occur
    """
    dict_keys = list(input_dict.keys())
    dict_keys.sort()
    dict_values = [input_dict[key] for key in dict_keys]

    def step_function(argument):
        if argument >= dict_keys[-1]:
            return dict_values[-1]
        else:
            for key in range(len(dict_keys)):
                if argument < dict_keys[key + 1]:
                    return dict_values[key]

    return step_function


def sinusoidal_scaling_function(start_time, baseline_value, end_time, final_value):
    """
    with a view to implementing scale-up functions over time, use the cosine function to produce smooth scale-up
    functions from one point to another
    """
    def sinusoidal_function(x):
        if not isinstance(x, float):
            raise ValueError("value fed into scaling function not a float")
        elif start_time > end_time:
            raise ValueError("start time is later than end time")
        elif x < start_time:
            return baseline_value
        elif start_time <= x <= end_time:
            return baseline_value + \
                   (final_value - baseline_value) * \
                   (0.5 - 0.5 * numpy.cos((x - start_time) * numpy.pi / (end_time - start_time)))
        else:
            return final_value
    return sinusoidal_function


def logistic_scaling_function(parameter):
    """
    a specific sigmoidal form of function that scales up from zero to one around the point of parameter
    won't be useful in all situations and is specifically for age-specific infectiousness - should be the same as in
    Romain's BMC Medicine manuscript
    """
    def sigmoidal_function(x):
        return 1.0 - 1.0 / (1.0 + numpy.exp(-(parameter - x)))

    return sigmoidal_function


def get_average_value_function(input_function, start_value, end_value):
    """
    use numeric integration to find the average value of a function between two extremes
    """
    return quad(input_function, start_value, end_value)[0] / (end_value - start_value)


def get_parameter_dict_from_function(input_function, breakpoints, upper_value=100.0):
    """
    create a dictionary of parameter values from a continuous function, an arbitrary upper value and some breakpoints
    within which to evaluate the function
    """
    breakpoints_with_upper = copy.copy(breakpoints)
    breakpoints_with_upper.append(upper_value)
    param_values = []
    for param_breakpoints in range(len(breakpoints)):
        param_values.append(get_average_value_function(
            input_function, breakpoints_with_upper[param_breakpoints], breakpoints_with_upper[param_breakpoints + 1]))
    return {str(key): value for key, value in zip(breakpoints, param_values)}


def store_database(outputs, table_name='outputs'):
    """
    store outputs from the model in sql database for use in producing outputs later
    """
    engine = create_engine("sqlite:///../databases/outputs.db", echo=True)
    if table_name == 'functions':
        outputs.to_sql(table_name, con=engine, if_exists="replace", index=False, dtype={"cdr_values": FLOAT()})
    else:
        outputs.to_sql(table_name, con=engine, if_exists="replace", index=False)


def create_flowchart(model_object, strata=-1, stratify=True, name="flow_chart"):
    """
    use graphviz module to create flow diagram of compartments and intercompartmental flows.
    """

    # set styles for graph
    styles = {"graph": {"label": "",
                        "fontsize": "16", },
              "nodes": {"fontname": "Helvetica",
                        "style": "filled",
                        "fillcolor": "#CCDDFF", },
              "edges": {"style": "dotted",
                        "arrowhead": "open",
                        "fontname": "Courier",
                        "fontsize": "10", }}

    def apply_styles(graph, styles):
        graph.graph_attr.update(("graph" in styles and styles["graph"]) or {})
        graph.node_attr.update(("nodes" in styles and styles["nodes"]) or {})
        graph.edge_attr.update(("edges" in styles and styles["edges"]) or {})
        return graph

    # find input nodes and edges
    if stratify:
        input_nodes = model_object.compartment_names
        type_of_flow = model_object.transition_flows[model_object.transition_flows.implement == strata] if strata != -1 else \
            model_object.transition_flows[model_object.transition_flows.implement == len(model_object.strata)]
    else:
        input_nodes = model_object.compartment_types
        type_of_flow = model_object.unstratified_flows

    # inputs are sectioned according to the stem value so colours can be added to each type, not yet implemented
    broken_down_nodes = []
    for stem_value in range(len(model_object.compartment_types)):
        x_vector = []
        for stem_type in range(len(input_nodes)):
            if model_object.compartment_types[stem_value] == find_stem(input_nodes[stem_type]):
                x_vector.append(input_nodes[stem_type])
        broken_down_nodes.append(stem_value)
        broken_down_nodes[stem_value] = x_vector

    model_object.flow_diagram = Digraph(format="png")
    if strata != -1:

        # find the compartment names that will be used to make the graph
        new_labels = list(set().union(type_of_flow["origin"].values, type_of_flow["to"].values))
        node_color = '#F0FFFF'
        for label in new_labels:
            if label in broken_down_nodes[0]:
                node_color = '#F0FFFF'
            if label in broken_down_nodes[1]:
                node_color = '#FAEBD7'
            if label in broken_down_nodes[2]:
                node_color = '#E9967A'
            model_object.flow_diagram.node(label, fillcolor=node_color)
    else:
        for label in model_object.compartment_names:
            node_color = '#F0FFFF'
            if label in broken_down_nodes[0]:
                node_color = '#F0FFFF'
            if label in broken_down_nodes[1]:
                node_color = '#FAEBD7'
            if label in broken_down_nodes[2]:
                node_color = '#E9967A'
            model_object.flow_diagram.node(label, fillcolor=node_color)

    # build the graph edges
    for row in type_of_flow.iterrows():
        model_object.flow_diagram.edge(row[1]["origin"], row[1]["to"], row[1]["parameter"])
    model_object.flow_diagram = apply_styles(model_object.flow_diagram, styles)
    model_object.flow_diagram.render(name)


class EpiModel:
    """
    general epidemiological model for constructing compartment-based models, typically of infectious disease
    transmission. See README.md for full description of purpose and approach of this model.

    :attribute times: list
        time steps at which outputs are to be evaluated
    :attribute compartment_types: list
        strings representing the compartments of the model
    :attribute initial_conditions: dict
        keys are compartment types, values are starting population values for each compartment
        note that not all compartment_types must be included as keys here
    :attribute parameters: dict
        constant parameter values
    :attribute requested_flows: list of dicts in standard format
        list with each element being a model flow, with fixed key names according to the type of flow implemented
    :attribute initial_conditions_to_total: bool
        whether to add the initial conditions up to a certain total if this value hasn't yet been reached through
        the initial_conditions argument
    :attribute infectious_compartment: str
        name of the infectious compartment for calculation of intercompartmental infection flows
    :attribute birth_approach: str
        approach to allowing entry flows into the model, must be add_crude_birth_rate, replace_deaths or no_births
    :attribute verbose: bool
        whether to output progress in model construction as this process proceeds
    :attribute reporting_sigfigs: int
        number of significant figures to output to when reporting progress
    :attribute entry_compartment: str
        name of the compartment that births come in to
    :attribute starting_population: numeric
        value for the total starting population to be supplemented to if initial_conditions_to_total requested
    :attribute starting_compartment: str
        optional name of the compartment to add population recruitment to
    :attribute equilibrium_stopping_tolerance: float
        value at which relative changes in compartment size trigger stopping when equilibrium reached
    :attribute integration_type: str
        integration approach for numeric solution to odes, must be odeint or solveivp currently
    """

    """
    most general methods
    """

    def find_parameter_value(self, parameter_name, time):
        """
        find the value of a parameter with time-variant values trumping constant ones

        :param parameter_name: string for the name of the parameter of interest
        :param time: model integration time (if needed)
        :return: parameter value, whether constant or time variant
        """
        return self.time_variants[parameter_name](time) if parameter_name in self.time_variants \
            else self.parameters[parameter_name]

    def output_to_user(self, comment):
        """
        short function to save the if statement in every call to output some information, may be adapted later and was
        more important to the R version of the repository

        :param: comment: string for the comment to be displayed to the user
        """
        if self.verbose:
            print(comment)

    """
    model construction methods
    """

    def __init__(self, times, compartment_types, initial_conditions, parameters, requested_flows,
                 initial_conditions_to_total=True, infectious_compartment="infectious", birth_approach="no_birth",
                 verbose=False, reporting_sigfigs=4, entry_compartment="susceptible", starting_population=1,
                 starting_compartment="", equilibrium_stopping_tolerance=1e-6, integration_type="odeint",
                 output_connections={}):
        """
        construction method to create a basic (and at this stage unstratified) compartmental model, including checking
        that the arguments have been provided correctly (in a separate method called here)

        :params: all arguments essentially become object attributes and are described in the first main docstring to
            this object class
        """

        # set flow attributes as pandas dataframes with fixed column names
        self.transition_flows = pd.DataFrame(columns=("type", "parameter", "origin", "to", "implement"))
        self.death_flows = pd.DataFrame(columns=("type", "parameter", "origin", "implement"))

        # attributes with specific format that are independent of user inputs
        self.tracked_quantities, self.time_variants = ({} for _ in range(2))
        self.derived_outputs = {"times": []}
        self.compartment_values, self.compartment_names, self.all_stratifications = ([] for _ in range(3))

        # ensure requests are fed in correctly
        self.check_and_report_attributes(
            times, compartment_types, initial_conditions, parameters, requested_flows, initial_conditions_to_total,
            infectious_compartment, birth_approach, verbose, reporting_sigfigs, entry_compartment,
            starting_population, starting_compartment, equilibrium_stopping_tolerance, integration_type,
            output_connections)

        # stop ide complaining about attributes being defined outside __init__, even though they aren't
        self.times, self.compartment_types, self.initial_conditions, self.parameters, self.requested_flows, \
            self.initial_conditions_to_total, self.infectious_compartment, self.birth_approach, self.verbose, \
            self.reporting_sigfigs, self.entry_compartment, self.starting_population, \
            self.starting_compartment, self.default_starting_population, self.equilibrium_stopping_tolerance, \
            self.unstratified_flows, self.outputs, self.integration_type, self.flow_diagram, self.output_connections = \
            (None for _ in range(20))

        # convert input arguments to model attributes
        for attribute in \
                ["times", "compartment_types", "initial_conditions", "parameters", "initial_conditions_to_total",
                 "infectious_compartment", "birth_approach", "verbose", "reporting_sigfigs", "entry_compartment",
                 "starting_population", "starting_compartment", "infectious_compartment",
                 "equilibrium_stopping_tolerance", "integration_type", "output_connections"]:
            setattr(self, attribute, eval(attribute))

        # set initial conditions and implement flows
        self.set_initial_conditions(initial_conditions_to_total)

        # implement unstratified flows
        self.implement_flows(requested_flows)

        # add any missing quantities that will be needed
        self.add_default_quantities()

    def check_and_report_attributes(
            self, _times, _compartment_types, _initial_conditions, _parameters, _requested_flows,
            _initial_conditions_to_total, _infectious_compartment, _birth_approach, _verbose, _reporting_sigfigs,
            _entry_compartment, _starting_population, _starting_compartment, _equilibrium_stopping_tolerance,
            _integration_type, _output_connections):
        """
        check all input data have been requested correctly

        :parameters: all parameters have come directly from the construction (__init__) method unchanged and have been
            renamed with a preceding _ character
        """

        # check that variables are of the expected type
        for expected_numeric_variable in ["_reporting_sigfigs", "_starting_population"]:
            if not isinstance(eval(expected_numeric_variable), int):
                raise TypeError("expected integer for %s" % expected_numeric_variable)
        for expected_float_variable in ["_equilibrium_stopping_tolerance"]:
            if not isinstance(eval(expected_float_variable), float):
                raise TypeError("expected float for %s" % expected_float_variable)
        for expected_list in ["_times", "_compartment_types", "_requested_flows"]:
            if not isinstance(eval(expected_list), list):
                raise TypeError("expected list for %s" % expected_list)
        for expected_string in \
                ["_infectious_compartment", "_birth_approach", "_entry_compartment", "_starting_compartment",
                 "_integration_type"]:
            if not isinstance(eval(expected_string), str):
                raise TypeError("expected string for %s" % expected_string)
        for expected_boolean in ["_initial_conditions_to_total", "_verbose"]:
            if not isinstance(eval(expected_boolean), bool):
                raise TypeError("expected boolean for %s" % expected_boolean)

        # check some specific requirements
        if _infectious_compartment not in _compartment_types:
            ValueError("infectious compartment name is not one of the listed compartment types")
        if _birth_approach not in ("add_crude_birth_rate", "replace_deaths", "no_births"):
            ValueError("requested birth approach unavailable")
        if sorted(_times) != _times:
            self.output_to_user("requested integration times are not sorted, now sorting")
            self.times = sorted(self.times)
        for output in _output_connections:
            if any(item not in ("origin", "to") for item in _output_connections[output]):
                raise ValueError("output connections incorrect specified, need an 'origin' and possibly a 'to' key")

        # report on characteristics of inputs
        if _verbose:
            print("integration times are from %s to %s (time units are always arbitrary)"
                  % (round(_times[0], _reporting_sigfigs), round(_times[-1], _reporting_sigfigs)))
            print("unstratified requested initial conditions are:")
            for compartment in _initial_conditions:
                print("\t%s: %s" % (compartment, _initial_conditions[compartment]))
            print("infectious compartment is called '%s'" % _infectious_compartment)
            print("birth approach is %s" % _birth_approach)

    def set_initial_conditions(self, _initial_conditions_to_total):
        """
        set starting compartment values

        :param _initial_conditions_to_total: bool
            unchanged from argument to __init__
        """

        # keep copy of the compartment types for when the compartment names are stratified later
        self.compartment_names = copy.copy(self.compartment_types)

        # start from making sure all compartments are set to zero values
        self.compartment_values = [0.0] * len(self.compartment_names)

        # set starting values of unstratified compartments to requested value
        for compartment in self.initial_conditions:
            if compartment in self.compartment_types:
                self.compartment_values[self.compartment_names.index(compartment)] = \
                    self.initial_conditions[compartment]
            else:
                raise ValueError("compartment %s requested in initial conditions not found in model compartment types")

        # sum to a total value if requested
        if _initial_conditions_to_total:
            self.sum_initial_compartments_to_total()

    def sum_initial_compartments_to_total(self):
        """
        make initial conditions sum to a certain value
        """
        compartment = self.find_remainder_compartment()
        remaining_population = self.starting_population - sum(self.compartment_values)
        if remaining_population < 0.0:
            raise ValueError("total of requested compartment values is greater than the requested starting population")
        self.output_to_user("requested that total population sum to %s" % self.starting_population)
        self.output_to_user("remaining population of %s allocated to %s compartment"
                            % (remaining_population, compartment))
        self.compartment_values[self.compartment_names.index(compartment)] = remaining_population

    def find_remainder_compartment(self):
        """
        find the compartment to put the remaining population that hasn't been assigned yet when summing to total

        :return: str
            name of the compartment to assign the remaining population size to
        """
        if len(self.starting_compartment) > 0 and \
                self.starting_compartment not in self.compartment_types:
            raise ValueError("starting compartment to populate with initial values not found in available compartments")
        elif len(self.starting_compartment) > 0:
            return self.starting_compartment
        else:
            self.output_to_user("no default starting compartment requested for unallocated population, " +
                                "so will be allocated to entry compartment %s" % self.entry_compartment)
            return self.entry_compartment

    def implement_flows(self, _requested_flows):
        """
        add all flows to create data frames from input lists

        :param _requested_flows: dict
            unchanged from argument to __init__
        """

        for flow in _requested_flows:

            # check flow requested correctly
            if flow["parameter"] not in self.parameters:
                raise ValueError("flow parameter not found in parameter list")
            if flow["origin"] not in self.compartment_types:
                raise ValueError("from compartment name not found in compartment types")
            if "to" in flow and flow["to"] not in self.compartment_types:
                raise ValueError("to compartment name not found in compartment types")

            # add flow to appropriate dataframe
            if flow["type"] == "compartment_death":
                self.add_death_flow(flow)
            else:
                self.add_transition_flow(flow)

            # add any tracked quantities that will be needed for calculating flow rates during integration
            if "infection" in flow["type"]:
                self.tracked_quantities["infectious_population"] = 0.0
            if flow["type"] == "infection_frequency":
                self.tracked_quantities["total_population"] = 0.0

    def add_default_quantities(self):
        """
        add parameters and tracked quantities that weren't requested but will be needed
        """

        # universal death rate
        if "universal_death_rate" not in self.parameters:
            self.parameters["universal_death_rate"] = 0.0

        # birth approach-specific parameters
        if self.birth_approach == "add_crude_birth_rate" and "crude_birth_rate" not in self.parameters:
            self.parameters["crude_birth_rate"] = 0.0
        elif self.birth_approach == "replace_deaths":
            self.tracked_quantities["total_deaths"] = 0.0

        # for each derived output to be recorded, initialise a tracked quantities key to zero
        for output in self.output_connections:
            self.tracked_quantities[output] = 0.0
            self.derived_outputs[output] = []

        # parameters essential for stratification
        self.parameters["entry_fractions"] = 1.0

    def add_transition_flow(self, _flow):
        """
        add a flow (row) to the dataframe storing the flows
        """

        # implement value starts at zero for unstratified and is then progressively incremented
        _flow["implement"] = 0
        self.transition_flows = self.transition_flows.append(_flow, ignore_index=True)

    def add_death_flow(self, _flow):
        """
        similarly for compartment-specific death flows
        """
        _flow["implement"] = 0
        self.death_flows = self.death_flows.append(_flow, ignore_index=True)

    """
    methods for model running
    """

    def run_model(self):
        """
        main function to integrate model odes, called externally in the master running script
        """
        self.output_to_user("now integrating")
        self.prepare_stratified_parameter_calculations()

        # basic default integration method
        if self.integration_type == "odeint":
            def make_model_function(compartment_values, time):
                self.update_tracked_quantities(compartment_values)
                return self.apply_all_flow_types_to_odes([0.0] * len(self.compartment_names), compartment_values, time)
            self.outputs = odeint(make_model_function, self.compartment_values, self.times)

        # alternative integration method
        elif self.integration_type == "solve_ivp":

            # solve_ivp requires arguments to model function in the reverse order
            def make_model_function(time, compartment_values):
                self.update_tracked_quantities(compartment_values)
                return self.apply_all_flow_types_to_odes([0.0] * len(self.compartment_names), compartment_values, time)

            # add a stopping condition, which was the original purpose of using this integration approach
            def set_stopping_conditions(time, compartment_values):
                self.update_tracked_quantities(compartment_values)
                net_flows = \
                    self.apply_all_flow_types_to_odes([0.0] * len(self.compartment_names), compartment_values, time)
                return max(list(map(abs, net_flows))) - self.equilibrium_stopping_tolerance
            set_stopping_conditions.terminal = True

            # solve_ivp returns more detailed structure, with (transposed) outputs (called "y") being just one component
            self.outputs = solve_ivp(
                make_model_function,
                (self.times[0], self.times[-1]), self.compartment_values, t_eval=self.times,
                events=set_stopping_conditions)["y"].transpose()

        else:
            raise ValueError("integration approach requested not available")
        self.output_to_user("integration complete")

    def prepare_stratified_parameter_calculations(self):
        """
        for use in the stratified version only
        """
        pass

    def apply_all_flow_types_to_odes(self, _ode_equations, _compartment_values, _time):
        """
        apply all flow types to a vector of zeros (note deaths must come before births in case births replace deaths)

        :param _ode_equations: list
            comes in as a list of zeros with length equal to that of the compartments
        :param _compartment_values: numpy.ndarray
            working values of the compartment sizes
        :param _time:
            current integration time
        :return: ode equations as list
            updated ode equations in same format but with all flows implemented
        """
        _ode_equations = self.apply_transition_flows(_ode_equations, _compartment_values, _time)
        _ode_equations = self.apply_compartment_death_flows(_ode_equations, _compartment_values, _time)
        _ode_equations = self.apply_universal_death_flow(_ode_equations, _compartment_values, _time)
        return self.apply_birth_rate(_ode_equations, _compartment_values)

    def apply_transition_flows(self, _ode_equations, _compartment_values, _time):
        """
        apply fixed or infection-related intercompartmental transition flows to odes

        :parameters and return: see previous method apply_all_flow_types_to_odes
        """
        for n_flow in self.transition_flows[self.transition_flows.implement == len(self.all_stratifications)].index:

            # find adjusted parameter value
            adjusted_parameter = self.get_parameter_value(self.transition_flows.parameter[n_flow], _time)

            # find from compartment and "infectious population" (which is 1 for standard flows)
            infectious_population = self.find_infectious_multiplier(self.transition_flows.type[n_flow])

            # calculate the n_flow and apply to the odes
            from_compartment = self.compartment_names.index(self.transition_flows.origin[n_flow])
            net_flow = adjusted_parameter * _compartment_values[from_compartment] * infectious_population
            _ode_equations = increment_compartment(_ode_equations, from_compartment, -net_flow)
            _ode_equations = increment_compartment(
                _ode_equations, self.compartment_names.index(self.transition_flows.to[n_flow]), net_flow)

            # track any quantities dependent on n_flow rates
            self.track_derived_outputs(n_flow, net_flow)

        # add another element to the derived outputs vector
        self.extend_derived_outputs(_time)

        # return n_flow rates
        return _ode_equations

    def track_derived_outputs(self, _n_flow, _net_flow):
        """
        calculate derived quantities to be tracked, which are stored as the self.derived_outputs dictionary for the
        current working time step

        :param _n_flow: int
            row number of the flow being considered in the preceding method
        :param _net_flow: float
            previously calculated magnitude of the transition flow
        """
        for output_type in self.output_connections:
            if self.output_connections[output_type]["origin"] in self.transition_flows.origin[_n_flow] \
                    and self.output_connections[output_type]["to"] in self.transition_flows.to[_n_flow]:
                self.tracked_quantities[output_type] += _net_flow

    def extend_derived_outputs(self, _time):
        """
        add the derived quantities being tracked to the end of the tracking vector, taking the self.derived_outputs
        dictionary for a single time point and updating the derived outputs dictionary of lists for all time points

        :param _time: float
            current time in integration process
        """
        self.derived_outputs["times"].append(_time)
        for output_type in self.output_connections:
            self.derived_outputs[output_type].append(self.tracked_quantities[output_type])

    def apply_compartment_death_flows(self, _ode_equations, _compartment_values, _time):
        """
        equivalent method to for transition flows above, but for deaths

        :parameters and return: see previous method apply_all_flow_types_to_odes
        """
        for n_flow in self.death_flows[self.death_flows.implement == len(self.all_stratifications)].index:
            adjusted_parameter = self.get_parameter_value(self.death_flows.parameter[n_flow], _time)
            from_compartment = self.compartment_names.index(self.death_flows.origin[n_flow])
            net_flow = adjusted_parameter * _compartment_values[from_compartment]
            _ode_equations = increment_compartment(_ode_equations, from_compartment, -net_flow)
            if "total_deaths" in self.tracked_quantities:
                self.tracked_quantities["total_deaths"] += net_flow
        return _ode_equations

    def apply_universal_death_flow(self, _ode_equations, _compartment_values, _time):
        """
        apply the population-wide death rate to all compartments

        :parameters and return: see previous method apply_all_flow_types_to_odes
        """
        for compartment in self.compartment_names:
            adjusted_parameter = self.get_parameter_value("universal_death_rate", _time)
            from_compartment = self.compartment_names.index(compartment)
            net_flow = adjusted_parameter * _compartment_values[from_compartment]
            _ode_equations = increment_compartment(_ode_equations, from_compartment, -net_flow)

            # track deaths in case births need to replace deaths
            if "total_deaths" in self.tracked_quantities:
                self.tracked_quantities["total_deaths"] += net_flow
        return _ode_equations

    def apply_birth_rate(self, _ode_equations, _compartment_values):
        """
        apply a birth rate to the entry compartments

        :parameters and return: see previous method apply_all_flow_types_to_odes
        """
        return increment_compartment(_ode_equations, self.compartment_names.index(self.entry_compartment),
                                     self.find_total_births(_compartment_values))

    def find_total_births(self, _compartment_values):
        """
        work out the total births to apply dependent on the approach requested

        :param _compartment_values:
            as for preceding methods
        :return: float
            total rate of births to be implemented in the model
        """
        if self.birth_approach == "add_crude_birth_rate":
            return self.parameters["crude_birth_rate"] * sum(_compartment_values)
        elif self.birth_approach == "replace_deaths":
            return self.tracked_quantities["total_deaths"]
        else:
            return 0.0

    def find_infectious_multiplier(self, flow_type):
        """
        find the multiplier to account for the infectious population in dynamic flows

        :param flow_type: str
            type of flow, as per the standard naming approach to flow types for the dataframes flow attribute
        :return:
            the total infectious quantity, whether that be the number or proportion of infectious persons
            needs to return as one for flows that are not transmission dynamic infectiousness flows
        """
        if flow_type == "infection_density":
            return self.tracked_quantities["infectious_population"]
        elif flow_type == "infection_frequency":
            return self.tracked_quantities["infectious_population"] / self.tracked_quantities["total_population"]
        else:
            return 1.0

    def update_tracked_quantities(self, _compartment_values):
        """
        update quantities that emerge during model running (not pre-defined functions of time)

        :param _compartment_values:
            as for preceding methods
        """
        for quantity in self.tracked_quantities:
            self.tracked_quantities[quantity] = 0.0
            if quantity == "infectious_population":
                self.find_infectious_population(_compartment_values)
            elif quantity == "total_population":
                self.tracked_quantities["total_population"] = sum(_compartment_values)

    def find_infectious_population(self, _compartment_values):
        """
        calculations to find the effective infectious population

        :param _compartment_values:
            as for preceding methods
        """
        for compartment in [comp for comp in self.compartment_names if find_stem(comp) == self.infectious_compartment]:
            self.tracked_quantities["infectious_population"] += \
                _compartment_values[self.compartment_names.index(compartment)]

    def get_parameter_value(self, _parameter, _time):
        """
        very simple, essentially place-holding, but need to split this out as a function in order to
        stratification later

        :param _parameter: str
            parameter name
        :param _time: float
            current integration time
        :return: float
            parameter value
        """
        return self.find_parameter_value(_parameter, _time)

    """
    simple output methods (most outputs will be managed outside of the python code)
    """

    def get_total_compartment_size(self, compartment_tags):
        """
        find the total values of the compartments listed by the user

        :param compartment_tags: list
            list of string variables for the compartment stems of interest
        """
        indices_to_plot = \
            [i for i in range(len(self.compartment_names)) if find_stem(self.compartment_names[i]) in compartment_tags]
        return self.outputs[:, indices_to_plot].sum(axis=1)

    def plot_compartment_size(self, compartment_tags, multiplier=1.):
        """
        plot the aggregate population of the compartments, the name of which contains all items of the list
        compartment_tags
        kept very simple for now, because output visualisation will generally be done outside of Python

        :param compartment_tags: list
            ilst of string variables for the compartments to plot
        :param multiplier: float
            scalar value to multiply the compartment values by
        """
        matplotlib.pyplot.plot(self.times, multiplier * self.get_total_compartment_size(compartment_tags))
        matplotlib.pyplot.show()


class StratifiedModel(EpiModel):
    """
    stratified version of the epidemiological model, inherits from EpiModel which is a concrete class and can run models
    independently (and could even include stratifications by using loops in a more traditional way to coding these
    models)

    :attribute all_stratifications: list
        all the stratification names implemented so far
    :attribute removed_compartments: list
        all unstratified compartments that have been removed through the stratification process
    :attribute overwrite_parameters: list
        any parameters that are intended as absolute values to be applied to that stratum and not multipliers for the
        unstratified parameter further up the tree
    :attribute compartment_types_to_stratify
        see check_compartment_request
    :attribute heterogeneous_infectiousness
    :attribute infectiousness_adjustments
    :attribute parameter_components
    """

    """
    most general methods
    """

    def add_compartment(self, new_compartment_name, new_compartment_value):
        """
        add a compartment by specifying its name and value to take

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
        store name of removed compartment in removed_compartments attribute

        :param compartment_name: str
            name of compartment to be removed
        """
        self.removed_compartments.append(compartment_name)
        del self.compartment_values[self.compartment_names.index(compartment_name)]
        del self.compartment_names[self.compartment_names.index(compartment_name)]
        self.output_to_user("removing compartment: %s" % compartment_name)

    def __init__(self, times, compartment_types, initial_conditions, parameters, requested_flows,
                 initial_conditions_to_total=True, infectious_compartment="infectious", birth_approach="no_birth",
                 verbose=False, reporting_sigfigs=4, entry_compartment="susceptible", starting_population=1,
                 starting_compartment="", equilibrium_stopping_tolerance=1e-6, integration_type="odeint",
                 output_connections={}):
        """
        constructor mostly inherits from parent class, with a few additional attributes that are required for the
        stratified version

        :parameters: all parameters coming in as arguments are those that are also attributes of the parent class
        """

        EpiModel.__init__(self, times, compartment_types, initial_conditions, parameters, requested_flows,
                          initial_conditions_to_total=initial_conditions_to_total,
                          infectious_compartment=infectious_compartment, birth_approach=birth_approach,
                          verbose=verbose, reporting_sigfigs=reporting_sigfigs, entry_compartment=entry_compartment,
                          starting_population=starting_population,
                          starting_compartment=starting_compartment,
                          equilibrium_stopping_tolerance=equilibrium_stopping_tolerance,
                          integration_type=integration_type, output_connections=output_connections)

        self.all_stratifications, self.removed_compartments, self.overwrite_parameters, \
        self.compartment_types_to_stratify, self.strata = \
            [[] for _ in range(5)]
        self.heterogeneous_infectiousness = False
        self.infectiousness_adjustments, self.parameter_components = [{} for _ in range(2)]

    """
    main master method for model stratification
    """

    def stratify(self, stratification_name, strata_request, compartment_types_to_stratify, adjustment_requests=(),
                 requested_proportions={}, infectiousness_adjustments=(), verbose=True):
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
        :param infectiousness_adjustments:

        :param verbose: bool
            whether to report on progress, note that this can be changed at this stage from what was requested at
            the original unstratified model construction
        """

        # check inputs correctly specified
        strata_names, adjustment_requests = self.prepare_and_check_stratification(
            stratification_name, strata_request, compartment_types_to_stratify, adjustment_requests, verbose)

        # work out ageing flows (comes first so that the compartment names are still in the unstratified form)
        if stratification_name == "age":
            self.set_ageing_rates(strata_names)

        # stratify the compartments
        requested_proportions = self.prepare_starting_proportions(strata_names, requested_proportions)
        self.stratify_compartments(stratification_name, strata_names, requested_proportions)

        # stratify the flows
        self.stratify_transition_flows(stratification_name, strata_names, adjustment_requests)
        self.stratify_entry_flows(stratification_name, strata_names, requested_proportions)
        if self.death_flows.shape[0] > 0:
            self.stratify_death_flows(stratification_name, strata_names, adjustment_requests)
        self.stratify_universal_death_rate(stratification_name, strata_names, adjustment_requests)

        # heterogeneous infectiousness adjustments
        self.apply_heterogeneous_infectiousness(stratification_name, strata_request, infectiousness_adjustments)

    """
    other pre-integration methods
    """

    def prepare_and_check_stratification(self, _stratification_name, _strata_request, _compartment_types_to_stratify,
                                         _adjustment_requests, _verbose):
        """
        initial preparation and checks

        :param _stratification_name: str
            the name of the stratification - i.e. the reason for implementing this type of stratification
        :param _strata_request:
            see find_strata_names_from_input
        :param _compartment_types_to_stratify:
            see check_compartment_request
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        :param _verbose:
            see stratify
        :return:
            _strata_names: list
                revised version of _strata_request after adaptation to class requirements
            adjustment_requests:
                revised version of _adjustment_requests after adaptation to class requirements
        """
        self.verbose = _verbose
        self.output_to_user("\nimplementing stratification for: %s" % _stratification_name)
        if _stratification_name == "age":
            _strata_request = self.check_age_stratification(_strata_request, _compartment_types_to_stratify)

        # make sure the stratification name is a string
        if type(_stratification_name) != str:
            _stratification_name = str(_stratification_name)
            self.output_to_user("converting stratification name %s to string" % _stratification_name)

        # ensure requested stratification hasn't previously been implemented
        if _stratification_name in self.all_stratifications:
            raise ValueError("requested stratification has already been implemented, please choose a different name")

        # record stratification as model attribute, find the names to apply strata and check requests
        self.all_stratifications.append(_stratification_name)
        _strata_names = self.find_strata_names_from_input(_strata_request)
        _adjustment_requests = self.incorporate_alternative_overwrite_approach(_adjustment_requests)
        self.check_compartment_request(_compartment_types_to_stratify)
        _adjustment_requests = self.check_parameter_adjustment_requests(_adjustment_requests, _strata_names)
        return _strata_names, _adjustment_requests

    def check_age_stratification(self, _strata_request, _compartment_types_to_stratify):
        """
        check that request meets the requirements for stratification by age

        :parameters: all parameters have come directly from the stratification (stratify) method unchanged and have been
            renamed with a preceding _ character
        :return: _strata_request: list
            revised names of the strata tiers to be implemented
        """
        self.output_to_user("implementing age stratification with specific behaviour")
        if len(_compartment_types_to_stratify) > 0:
            raise ValueError("requested age stratification, but compartment request should be passed as empty vector " +
                             "in order to apply to all compartments")
        elif any([type(stratum) != int and type(stratum) != float for stratum in _strata_request]):
            raise ValueError("inputs for age strata breakpoints are not numeric")
        elif "age" in self.strata:
            raise ValueError(
                "requested stratification by age, but this has specific behaviour and can only be applied once")
        if _strata_request != sorted(_strata_request):
            _strata_request = sorted(_strata_request)
            self.output_to_user("requested age strata not ordered, so have been sorted to: %s" % _strata_request)
        if 0 not in _strata_request:
            self.output_to_user("adding age stratum called '0' as not requested, to represent those aged less than %s"
                                % min(_strata_request))
            _strata_request.append(0)
        return _strata_request

    def find_strata_names_from_input(self, _strata_request):
        """
        find the names of the strata to be implemented from a particular user request

        :parameters: list or alternative format to be adapted
            strata requested in the format provided by the user (except for age, which is dealth with in the preceding
            method)
        :return: strata_names: list
            modified list of strata to be implemented in model
        """
        if type(_strata_request) == int:
            strata_names = numpy.arange(1, _strata_request + 1)
            self.output_to_user("single integer provided as strata labels for stratification, hence strata " +
                                "implemented are integers from 1 to %s" % _strata_request)
        elif type(_strata_request) == float:
            raise ValueError("single number passed as request for strata labels, but not an integer greater than " +
                             "one, so unclear what to do - therefore stratification failed")
        elif type(_strata_request) == list and len(_strata_request) > 0:
            strata_names = _strata_request
        else:
            raise ValueError("requested to stratify, but strata level names not submitted in correct format")
        for name in range(len(strata_names)):
            strata_names[name] = str(strata_names[name])
            self.output_to_user("adding stratum: %s" % strata_names[name])
        return strata_names

    def check_compartment_request(self, _compartment_types_to_stratify):
        """
        check the requested compartments to be stratified has been requested correctly

        :param _compartment_types_to_stratify: list
            the names of the compartment types that the requested stratification is intended to apply to
        """

        # if list of length zero passed, stratify all the compartment types in the model
        if len(_compartment_types_to_stratify) == 0:
            self.output_to_user("no compartment names specified for this stratification, " +
                                "so stratification applied to all model compartments")
            self.compartment_types_to_stratify = self.compartment_types

        # otherwise check all the requested compartments are available and implement the user request
        elif any([compartment not in self.compartment_types for compartment in self.compartment_types_to_stratify]):
            raise ValueError("requested compartment or compartments to be stratified are not available in this model")
        else:
            self.compartment_types_to_stratify = _compartment_types_to_stratify

    def incorporate_alternative_overwrite_approach(self, _adjustment_requests):
        """
        alternative approach to working out which parameters to overwrite
        can now put a capital W at the string's end to indicate that it is an overwrite parameter, as an alternative to
        submitting a separate dictionary key to represent the strata which need to be overwritten

        :param _adjustment_requests: dict
            user-submitted version of adjustment requests
        :return: revised_adjustments: dict
            modified version of _adjustment_requests after working out whether any parameters began with W
        """

        # has to be constructed as a separate dictionary to avoid it changing size during iteration
        revised_adjustments = {}
        for parameter in _adjustment_requests:

            # accept the key representing the overwrite parameters
            revised_adjustments[parameter] = {}
            revised_adjustments[parameter]["overwrite"] = \
                _adjustment_requests[parameter]["overwrite"] if "overwrite" in _adjustment_requests[parameter] else []

            # then loop through all the other keys of the user request
            for stratum in _adjustment_requests[parameter]:
                if stratum == "overwrite":
                    continue

                # if the parameter ends in W, it is interpreted as an overwrite parameter and added to this key
                elif stratum[-1] == "W":
                    revised_adjustments[parameter][stratum[: -1]] = _adjustment_requests[parameter][stratum]
                    revised_adjustments[parameter]["overwrite"].append(stratum[: -1])

                # otherwise just accept the parameter in its submitted form
                else:
                    revised_adjustments[parameter][stratum] = _adjustment_requests[parameter][stratum]
            _adjustment_requests[parameter] = revised_adjustments[parameter]
        return revised_adjustments

    def check_parameter_adjustment_requests(self, _adjustment_requests, _strata_names):
        """
        check parameter adjustments have been requested appropriately and add parameter for any strata not referred to

        :param _adjustment_requests: dict
            version of the submitted adjustment_requests modified by incorporate_alternative_overwrite_approach
        :param _strata_names:
            see find_strata_names_from_input
        :return: _adjustment_requests
            modified version of _adjustment_requests after checking
        """
        for parameter in _adjustment_requests:

            # check all the requested strata for parameter adjustments were strata that were requested
            if any([requested_stratum not in _strata_names + ["overwrite"]
                    for requested_stratum in _adjustment_requests[parameter]]):
                raise ValueError("stratum requested in adjustments but unavailable")

            # if any strata were not requested, assume a value of one
            for stratum in _strata_names:
                if stratum not in _adjustment_requests[parameter]:
                    _adjustment_requests[parameter][stratum] = 1
                    self.output_to_user("no request made for adjustment to %s within stratum " % parameter +
                                        "%s so accepting parent value by default" % stratum)
        return _adjustment_requests

    def prepare_starting_proportions(self, _strata_names, _requested_proportions):
        """
        prepare user inputs for starting proportions as needed
        must be specified with names that are strata being implemented during this stratification process
        note this applies to initial conditions and to entry flows

        :param _strata_names:
            see find_strata_names_from_input
        :param _requested_proportions: dict
            dictionary with keys for the stratum to assign starting population to and values the proportions to assign
        :return: dict
            revised dictionary of starting proportions after cleaning
        """
        if not all(stratum in _strata_names for stratum in _requested_proportions):
            raise ValueError("requested starting proportion for stratum that does not appear in requested strata")
        if any(_requested_proportions[stratum] > 1.0 for stratum in _requested_proportions):
            raise ValueError("requested a starting proportion value of greater than one")

        # assuming an equal proportion of the total for the compartment if not otherwise specified
        for stratum in _strata_names:
            if stratum not in _requested_proportions:
                starting_proportion = 1.0 / len(_strata_names)
                _requested_proportions[stratum] = starting_proportion
                self.output_to_user("no starting proportion requested for stratum %s" % stratum +
                                    " so allocated %s of total" % round(starting_proportion, self.reporting_sigfigs))

        # normalise the dictionary before return, in case adding the missing groups as equal proportions exceeds one
        return normalise_dict(_requested_proportions)

    def stratify_compartments(self, _stratification_name, _strata_names, _requested_proportions):
        """
        stratify the model compartments, which affects the compartment_names and the compartment_values attributes

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _requested_proportions:
            see prepare_starting_proportions
        """

        # find the existing compartments that need stratification
        for compartment in \
                [comp for comp in self.compartment_names if find_stem(comp) in self.compartment_types_to_stratify]:

            # add and remove compartments
            for stratum in _strata_names:
                self.add_compartment(create_stratified_name(compartment, _stratification_name, stratum),
                                     self.compartment_values[self.compartment_names.index(compartment)] *
                                     _requested_proportions[stratum])
            self.remove_compartment(compartment)

    def stratify_transition_flows(self, _stratification_name, _strata_names, _adjustment_requests):
        """
        stratify flows depending on whether inflow, outflow or both need replication, using call to add_stratified_flows
        method below

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        """
        for n_flow in self.transition_flows[self.transition_flows.implement == len(self.all_stratifications) - 1].index:
            self.add_stratified_flows(
                n_flow, _stratification_name, _strata_names,
                find_stem(self.transition_flows.origin[n_flow]) in self.compartment_types_to_stratify,
                find_stem(self.transition_flows.to[n_flow]) in self.compartment_types_to_stratify,
                _adjustment_requests)
        self.output_to_user("stratified transition flows matrix:\n%s" % self.transition_flows)

    def stratify_entry_flows(self, _stratification_name, _strata_names, _requested_proportions):
        """
        stratify entry/recruitment/birth flows according to requested entry proportion adjustments
        note this applies to initial conditions and to entry flows

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _requested_proportions:
            see prepare_starting_proportions
        :return:
            normalised dictionary of the compartments that the new entry flows should come in to
        """
        entry_fractions = {}
        if self.entry_compartment in self.compartment_types_to_stratify:
            for stratum in _strata_names:
                entry_fraction_name = create_stratified_name("entry_fraction", _stratification_name, stratum)

                # specific behaviour for age stratification
                if _stratification_name == "age" and str(stratum) == "0":
                    entry_fractions[entry_fraction_name] = 1.0
                    continue
                elif _stratification_name == "age":
                    entry_fractions[entry_fraction_name] = 0.0
                    continue

                # where a request has been submitted
                elif stratum in _requested_proportions:
                    entry_fractions[entry_fraction_name] = _requested_proportions[stratum]
                    self.output_to_user("assigning specified proportion of starting population to %s" % stratum)

                # otherwise if no request made
                else:
                    entry_fractions[entry_fraction_name] = 1.0 / len(_strata_names)
                    self.output_to_user("assuming %s " % entry_fractions[entry_fraction_name] +
                                        "of starting population to be assigned to %s stratum by default" % stratum)

        # normalise at the end before return
        self.parameters.update(normalise_dict(entry_fractions))

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
        for flow in self.death_flows[self.death_flows.implement == len(self.all_stratifications) - 1].index:
            for stratum in _strata_names:

                # get stratified parameter name if requested to stratify, otherwise use the unstratified one
                parameter_name = self.add_adjusted_parameter(
                    self.death_flows.parameter[flow], _stratification_name, stratum, _adjustment_requests)
                if not parameter_name:
                    parameter_name = self.death_flows.parameter[flow]

                # add the stratified flow
                self.death_flows = self.death_flows.append(
                    {"type": self.death_flows.type[flow],
                     "parameter": parameter_name,
                     "origin": create_stratified_name(self.death_flows.origin[flow], _stratification_name, stratum),
                     "implement": len(self.all_stratifications)},
                    ignore_index=True)

    def stratify_universal_death_rate(self, _stratification_name, _strata_names, _adjustment_requests):
        """
        stratify the approach to universal, population-wide deaths (which can be made to vary by stratum)
        adjust each parameter that refers to the universal death rate according to user request

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
             see find_strata_names_from_input
        :param _adjustment_requests:
             see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
       """
        for parameter in [param for param in self.parameters if find_stem(param) == "universal_death_rate"]:
            for stratum in _strata_names:
                self.add_adjusted_parameter(parameter, _stratification_name, stratum, _adjustment_requests)

    def add_adjusted_parameter(self, _unadjusted_parameter, _stratification_name, _stratum, _adjustment_requests):
        """
        find the adjustment request that is relevant to a particular unadjusted parameter and stratum
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

        # find the adjustment requests that are extensions of the base parameter type being considered
        for parameter_request in [req for req in _adjustment_requests if _unadjusted_parameter.startswith(req)]:
            parameter_adjustment_name = create_stratified_name(_unadjusted_parameter, _stratification_name, _stratum)
            self.output_to_user(
                "modifying %s for %s stratum of %s with new parameter called %s"
                % (_unadjusted_parameter, _stratum, _stratification_name, parameter_adjustment_name))

            # implement user request (otherwise parameter will be left out and assumed to be 1 during integration)
            if _stratum in _adjustment_requests[parameter_request]:
                self.parameters[parameter_adjustment_name] = _adjustment_requests[parameter_request][_stratum]

            # overwrite parameters higher up the tree by tracking which ones are to be overwritten
            if "overwrite" in _adjustment_requests[parameter_request] and \
                    _stratum in _adjustment_requests[parameter_request]["overwrite"]:
                self.overwrite_parameters.append(parameter_adjustment_name)
        return parameter_adjustment_name

    def apply_heterogeneous_infectiousness(self, stratification_name, strata_request, infectiousness_adjustments):
        """
        work out infectiousness adjustments and set as model attributes
        this has not been fully documented, as we are intending to revise this to permit any approach to heterogeneous
        infectiousness or mixing assumptions
        """
        if len(infectiousness_adjustments) == 0:
            self.output_to_user("heterogeneous infectiousness not requested for this stratification")
        elif self.infectious_compartment not in self.compartment_types_to_stratify:
            raise ValueError("request for infectiousness stratification does not apply to the infectious compartment")
        else:
            self.heterogeneous_infectiousness = True
            for stratum in infectiousness_adjustments:
                if stratum not in strata_request:
                    raise ValueError("stratum to have infectiousness modified not found within requested strata")
                adjustment_name = create_stratified_name("", stratification_name, stratum)
                self.infectiousness_adjustments[adjustment_name] = infectiousness_adjustments[stratum]

    def set_ageing_rates(self, _strata_names):
        """
        set intercompartmental flows for ageing from one stratum to the next as the reciprocal of the width of the age
        bracket

        :param _strata_names:
            see find_strata_names_from_input
        """
        for stratum_number in range(len(_strata_names[: -1])):
            start_age = int(_strata_names[stratum_number])
            end_age = int(_strata_names[stratum_number + 1])
            ageing_parameter_name = "ageing%sto%s" % (start_age, end_age)
            ageing_rate = 1.0 / (end_age - start_age)
            self.output_to_user("ageing rate from age group %s to %s is %s"
                                % (start_age, end_age, round(ageing_rate, self.reporting_sigfigs)))
            self.parameters[ageing_parameter_name] = ageing_rate
            for compartment in self.compartment_names:
                self.transition_flows = self.transition_flows.append(
                    {"type": "standard_flows",
                     "parameter": ageing_parameter_name,
                     "origin": create_stratified_name(compartment, "age", start_age),
                     "to": create_stratified_name(compartment, "age", end_age),
                     "implement": len(self.all_stratifications)},
                    ignore_index=True)

    def add_stratified_flows(self, _n_flow, _stratification_name, _strata_names, stratify_from, stratify_to,
                             _adjustment_requests):
        """
        add additional stratified flow to the transition flow data frame attribute of the class

        :param _n_flow: int
            location of the unstratified flow within the transition flow attribute
        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param stratify_from: bool
            whether to stratify the from/origin compartment
        :param stratify_to:
            whether to stratify the to/destination compartment
        :param _adjustment_requests:
            see incorporate_alternative_overwrite_approach and check_parameter_adjustment_requests
        """
        if stratify_from or stratify_to:
            self.output_to_user(
                "for flow from %s to %s in stratification %s"
                % (self.transition_flows.origin[_n_flow], self.transition_flows.to[_n_flow], _stratification_name))

            # loop over each stratum in the requested stratification structure
            for stratum in _strata_names:

                # find parameter name
                parameter_name = self.add_adjusted_parameter(
                    self.transition_flows.parameter[_n_flow], _stratification_name, stratum, _adjustment_requests)
                if not parameter_name:
                    parameter_name = self.sort_absent_parameter_request(
                        _stratification_name, _strata_names, stratum, stratify_from, stratify_to, _n_flow)
                self.output_to_user("\tadding parameter %s" % parameter_name)

                # determine whether to and/or from compartments are stratified
                from_compartment = \
                    create_stratified_name(self.transition_flows.origin[_n_flow], _stratification_name, stratum) if \
                        stratify_from else self.transition_flows.origin[_n_flow]
                to_compartment = \
                    create_stratified_name(self.transition_flows.to[_n_flow], _stratification_name, stratum) if \
                        stratify_to else self.transition_flows.to[_n_flow]

                # add the new flow
                self.transition_flows = self.transition_flows.append(
                    {"type": self.transition_flows.type[_n_flow],
                     "parameter": parameter_name,
                     "origin": from_compartment,
                     "to": to_compartment,
                     "implement": len(self.all_stratifications)},
                    ignore_index=True)

        # if flow applies to a transition not involved in the stratification, still increment to ensure implemented
        else:
            new_flow = self.transition_flows.loc[_n_flow, :].to_dict()
            new_flow["implement"] += 1
            self.transition_flows = self.transition_flows.append(new_flow, ignore_index=True)

    def sort_absent_parameter_request(self, _stratification_name, _strata_names, _stratum, _stratify_from, _stratify_to,
                                      _n_flow):
        """
        work out what to do if a specific parameter adjustment has not been requested

        :param _stratification_name:
            see prepare_and_check_stratification
        :param _strata_names:
            see find_strata_names_from_input
        :param _stratum:
        :param _stratify_from:
            see add_stratified_flows
        :param _stratify_to:
            see add_stratified_flows
        :param _n_flow: int
            index of the flow being dealt with
        :return: str
            parameter name for revised parameter than wasn't provided
        """

        # default behaviour if not specified is to split the parameter into equal parts if to compartment is split
        if not _stratify_from and _stratify_to:
            self.output_to_user("\tsplitting existing parameter value %s into %s equal parts"
                                % (self.transition_flows.parameter[_n_flow], len(_strata_names)))
            parameter_name = create_stratified_name(self.transition_flows.parameter[_n_flow], _stratification_name, _stratum)
            self.parameters[parameter_name] = 1.0 / len(_strata_names)

        # otherwise if no request, retain the existing parameter
        else:
            parameter_name = self.transition_flows.parameter[_n_flow]
            self.output_to_user("\tretaining existing parameter value %s" % parameter_name)
        return parameter_name

    def prepare_stratified_parameter_calculations(self):
        """
        prior to integration commencing, work out what the components are of each parameter being implemented
        """

        # create list of all the parameters that we need to find the set of adjustments for
        parameters_to_adjust = []
        for n_flow in range(self.transition_flows.shape[0]):
            if self.transition_flows.implement[n_flow] == len(self.all_stratifications) and \
                    self.transition_flows.parameter[n_flow] not in parameters_to_adjust:
                parameters_to_adjust.append(self.transition_flows.parameter[n_flow])
        for n_flow in range(self.death_flows.shape[0]):
            if self.death_flows.implement[n_flow] == len(self.all_stratifications) and \
                    self.death_flows.parameter[n_flow] not in parameters_to_adjust:
                parameters_to_adjust.append(self.death_flows.parameter[n_flow])
        parameters_to_adjust.append("universal_death_rate")

        # and adjust
        for parameter in parameters_to_adjust:
            self.find_parameter_components(parameter)

    def find_parameter_components(self, _parameter):
        """
        extract the components of the stratified parameter into a dictionary structure with values being a list of
        time-variant parameters, a list of constant parameters and the product of all the constant values applied

        :param _parameter: str
            name of the parameter that we are tracking down the components of
        """
        self.parameter_components[_parameter] = {"time_variants": [], "constants": [], "constant_value": 1}

        # work backwards through sub-strings of the parameter names from the full name to the name through to each X
        for x_instance in extract_reversed_x_positions(_parameter):
            component = _parameter[: x_instance]
            is_time_variant = component in self.time_variants
            if component in self.overwrite_parameters and is_time_variant:
                self.parameter_components[_parameter] = \
                    {"time_variants": [component], "constants": [], "constant_value": 1}
                break
            elif component in self.overwrite_parameters and not is_time_variant:
                self.parameter_components[_parameter] = \
                    {"time_variants": [], "constants": [component], "constant_value": 1}
                break
            elif is_time_variant:
                self.parameter_components[_parameter]["time_variants"].append(component)
            elif component in self.parameters:
                self.parameter_components[_parameter]["constants"].append(component)
            else:
                raise ValueError("unable to find parameter component %s of parameter %s" % (component, _parameter))

        # pre-calculate the constant component by multiplying through all the constant values
        for constant_parameter in self.parameter_components[_parameter]["constants"]:
            self.parameter_components[_parameter]["constant_value"] *= self.parameters[constant_parameter]

    """
    methods to be called during the process of model running
    """

    def get_parameter_value(self, _parameter, _time):
        """
        using the approach specified in find_parameter_components calculate adjusted parameter value from pre-calculated
        product of constant components and time variants

        :param _parameter: str
            name of the parameter whose value is needed
        :param _time: float
            time in model integration
        """
        adjusted_parameter = self.parameter_components[_parameter]["constant_value"]
        for time_variant in self.parameter_components[_parameter]["time_variants"]:
            adjusted_parameter *= self.time_variants[time_variant](_time)
        return adjusted_parameter

    def find_infectious_population(self, _compartment_values):
        """
        calculations to find the effective infectious population

        :param _compartment_values:
        """

        # loop through all compartments and find the ones representing active infectious disease
        for compartment in [comp for comp in self.compartment_names if find_stem(comp) == self.infectious_compartment]:

            # assume homogeneous infectiousness until/unless requested otherwise
            infectiousness_modifier = 1.0

            # haven't yet finished heterogeneous infectiousness - want to implement all forms of heterogeneous mixing
            if self.heterogeneous_infectiousness:
                for adjustment in [adj for adj in self.infectiousness_adjustments if adj in compartment]:
                    infectiousness_modifier = self.infectiousness_adjustments[adjustment]

            # update total infectious population
            self.tracked_quantities["infectious_population"] += \
                _compartment_values[self.compartment_names.index(compartment)] * infectiousness_modifier

    def apply_birth_rate(self, _ode_equations, _compartment_values):
        """
        apply a population-wide death rate to all compartments

        :parameters: all parameters have come directly from the apply_all_flow_types_to_odes method unchanged
        """
        total_births = self.find_total_births(_compartment_values)

        # split the total births across entry compartments
        for compartment in [comp for comp in self.compartment_names if find_stem(comp) == self.entry_compartment]:

            # calculate adjustment to original stem entry rate
            entry_fraction = 1.0
            x_positions = extract_x_positions(compartment)
            if len(x_positions) > 1:
                for x_instance in range(len(x_positions) - 1):
                    entry_fraction *= \
                        self.parameters["entry_fractionX%s"
                                        % compartment[x_positions[x_instance] + 1: x_positions[x_instance + 1]]]
            compartment_births = entry_fraction * total_births
            _ode_equations = increment_compartment(
                _ode_equations, self.compartment_names.index(compartment), compartment_births)
        return _ode_equations


if __name__ == "__main__":

    # example code to test out many aspects of SUMMER function - intended to be equivalent to the example in R
    sir_model = StratifiedModel(
        numpy.linspace(0, 60 / 365, 61).tolist(),
        ["susceptible", "infectious", "recovered"],
        {"infectious": 0.001},
        {"beta": 400, "recovery": 365 / 13, "infect_death": 1},
        [{"type": "standard_flows", "parameter": "recovery", "origin": "infectious", "to": "recovered"},
         {"type": "infection_density", "parameter": "beta", "origin": "susceptible", "to": "infectious"},
         {"type": "compartment_death", "parameter": "infect_death", "origin": "infectious"}],
        output_connections={"incidence": {"origin": "susceptible", "to": "infectious"}},
        verbose=False, integration_type="solve_ivp")
    sir_model.stratify("hiv", ["negative", "positive"], [],
                       {"recovery": {"negative": 0.7, "positive": 0.5},
                        "infect_death": {"negative": 0.5},
                        "entry_fraction": {"negative": 0.6, "positive": 0.4}},
                       {"negative": 0.6}, verbose=False)
    sir_model.stratify("age", [1, 10, 3], [], {"recovery": {"1": 0.5, "10": 0.8}}, verbose=False)

    sir_model.run_model()

    create_flowchart(sir_model, strata=len(sir_model.all_stratifications))

    sir_model.plot_compartment_size(['infectious', 'hiv_positive'])


