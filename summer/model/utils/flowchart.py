"""
Flow diagram creation
"""
from graphviz import Digraph
from summer.model.utils.string import find_name_components
from autumn import constants


def create_flowchart(model_object, strata=None, name="flow_chart"):
    """
    use graphviz module to create flow diagram of compartments and inter-compartmental flows

    :param model_object: summer object
        model whose inter-compartmental flows need to be graphed
    :param strata: int
        number of stratifications that have been implemented at the point that diagram creation requested
    :param name: str
        filename for the image to be put out as
    """

    # find the stratification level of interest, with the fully stratified model being the default
    if strata is None:
        strata = len(model_object.all_stratifications)

    # set styles for graph
    styles = {
        "graph": {"label": "", "fontsize": "16",},
        "nodes": {"fontname": "Helvetica", "style": "filled", "fillcolor": "#CCDDFF",},
        "edges": {"style": "dotted", "arrowhead": "open", "fontname": "Courier", "fontsize": "10",},
    }

    # colour dictionary for different nodes indicating different stages of infection
    default_colour_dict = {
        "susceptible": "#F0FFFF",
        "early_latent": "#A64942",
        "late_latent": "#A64942",
        "infectious": "#FE5F55",
        "recovered": "#FFF1C1",
    }

    def apply_styles(graph, _styles):
        graph.graph_attr.update(("graph" in _styles and _styles["graph"]) or {})
        graph.node_attr.update(("nodes" in _styles and _styles["nodes"]) or {})
        graph.edge_attr.update(("edges" in _styles and _styles["edges"]) or {})
        return graph

    # find input nodes and edges
    type_of_flow = model_object.transition_flows[model_object.transition_flows.implement == strata]

    # find compartment names to be used, from all compartments listed as origins or destinations in transition flows
    new_labels = list(set().union(type_of_flow["origin"].values, type_of_flow["to"].values))

    # start building graph
    model_object.flow_diagram = Digraph(format="png")

    # inputs are sectioned according to the stem value so colours can be added to each type
    for label in new_labels:
        comp_name = find_name_components(label)[0]
        node_color = (
            default_colour_dict[comp_name] if comp_name in default_colour_dict.keys() else "#F0FFFF"
        )
        model_object.flow_diagram.node(label, fillcolor=node_color)

    # build the graph edges
    for row in type_of_flow.iterrows():
        model_object.flow_diagram.edge(row[1]["origin"], row[1]["to"], row[1]["parameter"])
    model_object.flow_diagram = apply_styles(model_object.flow_diagram, styles)
    model_object.flow_diagram.render(directory=constants.DATA_DIR)
