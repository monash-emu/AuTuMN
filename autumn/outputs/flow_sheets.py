import os


def save_flows_sheets(model, out_dir):
    """
    Save the transitions and deaths dataframes as csv files, for easier inspection
    """
    model.transition_flows.to_csv(os.path.join(out_dir, "transitions.csv"))
    model.death_flows.to_csv(os.path.join(out_dir, "deaths.csv"))
