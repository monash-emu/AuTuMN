from typing import Dict, Union, Callable

from summer import CompartmentalModel


def broadcast_infection_flows_over_source(
    model: CompartmentalModel, 
    flow_name: str,
    source_comp: str,
    dest_comp: str,
    source_filter: Dict[str, str], 
    dest_filter: Dict[str, str], 
    contact_rate: Union[Callable, float],
    exp_flows: int,
):
    """
    Automatically broadcast all the flows over the different previously infected strata of a compartment that can be reinfected by strain.

    Args:
        model: The summer model object to be modified
        flow_name: The name of the flow to feed straight through
        source_comp: The source compartment to feed straight through
        dest_comp: The destination compartment to feed straight through
        source_filter: Any pre-existing source filtering being done on the flows
        dest_filter: Any pre-existing destination filtering being done on the flows
        contact_rate: The parameter value to feed straight through
        exp_flows: The expected number of flows to be created to feed straight through

    """

    # Loop over all source strain compartments
    for source_strain in model._disease_strains:
        strain_filter = {"strain": source_strain}
        source_filter.update(strain_filter)

        # Apply to model
        model.add_infection_frequency_flow(
            flow_name,
            contact_rate,
            source_comp,
            dest_comp,
            source_filter,
            dest_filter,
            expected_flow_count=exp_flows,
        )