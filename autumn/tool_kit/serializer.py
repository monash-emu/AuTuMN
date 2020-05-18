from summer.model import StratifiedModel


def serialize_model(model: StratifiedModel) -> dict:
    """
    Model serializer - transform all relevant model info into basic Python data structures
    """
    t_flows_df = model.transition_flows
    flow_implement = t_flows_df.implement.max()
    mask = t_flows_df["implement"] == flow_implement
    t_flows_imp_df = t_flows_df[mask]
    t_flows = list(t_flows_imp_df.T.to_dict().values())

    d_flows_df = model.death_flows
    flow_implement = d_flows_df.implement.max()
    mask = d_flows_df["implement"] == flow_implement
    d_flows_imp_df = d_flows_df[mask]
    d_flows = list(d_flows_imp_df.T.to_dict().values())
    return {
        "settings": {
            "entry_compartment": model.entry_compartment,
            "birth_approach": model.birth_approach,
            "infectious_compartment": model.infectious_compartment,
        },
        "infectiousness": {
            "infectiousness_levels": model.infectiousness_levels,
            "infectiousness_multipliers": model.infectiousness_multipliers,
        },
        "start": {
            "initial_conditions": model.initial_conditions,
            "starting_population": model.starting_population,
            "times": model.times,
        },
        "stratifications": model.all_stratifications,
        "parameters": model.parameters,
        "flows": {"transition": t_flows, "death": d_flows, "requested": model.requested_flows,},
        "adaptation_functions": list(model.adaptation_functions.keys()),
        "mixing": {
            "mixing_matrix": model.mixing_matrix.tolist(),
            "dynamic_mixing_matrix": model.dynamic_mixing_matrix,
        },
    }
