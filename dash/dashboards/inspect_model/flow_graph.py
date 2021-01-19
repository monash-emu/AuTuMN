import networkx as nx
import streamlit as st
from matplotlib import pyplot

from summer2.flows import (
    DeathFlow,
    BaseEntryFlow,
    BaseExitFlow,
    BaseTransitionFlow,
    BaseInfectionFlow,
)
from autumn.plots.plotter import StreamlitPlotter
from autumn.tool_kit.model_register import AppRegion

MARKERS = ".spP*D^vxH"
FLOW_STYLES = [
    {
        "class": DeathFlow,
        "color": "#f5424e",
        "style": "dashed",
        "alpha": 0.4,
    },
    {
        "class": BaseEntryFlow,
        "color": "green",
        "style": "dashed",
        "alpha": 0.4,
    },
    {
        "class": BaseTransitionFlow,
        "color": "#7a7eeb",
        "style": "solid",
        "alpha": 0.8,
    },
    {
        "class": BaseInfectionFlow,
        "color": "#f5424e",
        "style": "solid",
        "alpha": 0.8,
    },
]


def plot_flow_graph(plotter: StreamlitPlotter, app: AppRegion):
    """
    Plot a graph of the model's compartments and flows
    See NetworkX documentation: https://networkx.org/documentation/stable/index.html
    """
    model = app.build_model(app.params["default"])

    flow_types = st.multiselect(
        "Flow types", ["Transition", "Entry", "Exit"], default=["Transition"]
    )
    layout_lookup = {
        "Spring": nx.spring_layout,
        "Spectral": nx.spectral_layout,
        "Kamada Kawai": nx.kamada_kawai_layout,
        "Random": nx.random_layout,
    }
    layout_key = st.selectbox("Layout", list(layout_lookup.keys()))
    layout_func = layout_lookup[layout_key]

    is_node_labels_visible = st.checkbox("Show node labels")
    include_connected_nodes = st.checkbox("Include connected nodes")

    # ADD compartment selector
    original_compartment_names = model._original_compartment_names
    stratifications = model._stratifications
    compartment_names = model.compartments

    orig_comps = ["All"] + original_compartment_names
    chosen_comp_names = st.multiselect("Compartments", orig_comps, default="All")
    if "All" in chosen_comp_names:
        chosen_comp_names = original_compartment_names

    chosen_strata = {}
    for strat in stratifications:
        options = ["All"] + strat.strata
        choices = st.multiselect(strat.name, options, default=["All"])
        if "All" not in choices:
            chosen_strata[strat.name] = choices

    # Build the graph.
    comps_to_graph = []
    if "Entry" in flow_types:
        comps_to_graph.append("ENTRY")
    if "Exit" in flow_types:
        comps_to_graph.append("EXIT")

    for comp in compartment_names:
        is_selected = True
        if not comp.name in chosen_comp_names:
            is_selected = False

        for strat_name, strata in chosen_strata.items():
            has_strat = strat_name in comp._strat_names
            has_strata = any(comp.has_stratum(strat_name, s) for s in strata)
            if has_strat and not has_strata:
                is_selected = False
                continue

        if not is_selected:
            continue

        comps_to_graph.append(comp)

    if not comps_to_graph:
        st.write("Nothing to plot")
        return

    graph = nx.DiGraph()
    for comp in comps_to_graph:
        graph.add_node(comp)

    flow_lookup = {}
    for flow in model._flows:
        if "Entry" in flow_types and is_flow_type(flow, BaseEntryFlow):
            edge = ("ENTRY", flow.dest)
        elif "Exit" in flow_types and is_flow_type(flow, BaseExitFlow):
            edge = (flow.source, "EXIT")
        elif "Transition" in flow_types and is_flow_type(flow, BaseTransitionFlow):
            edge = (flow.source, flow.dest)
        else:
            continue

        if include_connected_nodes:
            src_is_valid = edge[0] in comps_to_graph and edge[0] != "ENTRY"
            dst_is_valid = edge[1] in comps_to_graph and edge[1] != "EXIT"
            should_add_edge = src_is_valid or dst_is_valid
        else:
            src_is_valid = edge[0] in comps_to_graph
            dst_is_valid = edge[1] in comps_to_graph
            should_add_edge = src_is_valid and dst_is_valid

        if should_add_edge:
            graph.add_edge(*edge)
            flow_lookup[edge_to_str(edge)] = flow

    # Draw the graph.
    pyplot.style.use("ggplot")
    fig = pyplot.figure(figsize=[14 / 1.5, 9 / 1.5], dpi=300)
    axis = fig.add_axes([0, 0, 1, 1])

    # Specify a layout technique.
    pos = layout_func(graph)

    # Draw the nodes.
    node_size = 12

    for node in graph.nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[node],
            ax=axis,
            node_size=node_size,
            node_color="#333",
            alpha=1,
        )

    if is_node_labels_visible:
        labels = {}
        for node in graph.nodes:
            labels[node] = get_label(node)

        nx.draw_networkx_labels(graph, pos, labels, font_size=8, verticalalignment="top", alpha=0.8)

    # Draw the edges between nodes.
    for edge in graph.edges:
        flow = flow_lookup[edge_to_str(edge)]
        color = "#666"
        style = "solid"
        alpha = 0.8

        for flow_style in FLOW_STYLES:
            if is_flow_type(flow, flow_style["class"]):
                color = flow_style["color"]
                style = flow_style["style"]
                alpha = flow_style["alpha"]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[edge],
            width=1,
            node_size=node_size,
            alpha=alpha,
            edge_color=color,
            style=style,
            arrowsize=6,
        )

    st.pyplot(fig)


def get_label(comp):

    if comp in ["ENTRY", "EXIT"]:
        return comp

    label = ""
    label += comp.name.replace("_", " ")
    for k, v in comp.strata.items():
        name = k.replace("_", " ")
        val = v.replace("_", " ")
        label += f"\n{name}: {val}"

    return label


def edge_to_str(edge):
    return str(edge[0]) + "|" + str(edge[1])


def is_flow_type(flow, flow_cls):
    return issubclass(flow.__class__, flow_cls)
