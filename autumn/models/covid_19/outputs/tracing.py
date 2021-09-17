from summer import CompartmentalModel


def request_tracing_outputs(model: CompartmentalModel):
    """
    Contact tracing-related outputs.
    """

    # Standard calculations always computed when contact tracing requested
    model.request_computed_value_output("prevalence")
    model.request_computed_value_output("prop_detected_traced")
    model.request_computed_value_output("prop_contacts_with_detected_index")
    model.request_computed_value_output("traced_flow_rate")

    # Proportion of quarantined contacts among all contacts
    model.request_function_output(
        name="prop_contacts_quarantined",
        func=lambda prop_detected_traced, prop_detected_index: prop_detected_traced * prop_detected_index,
        sources=["prop_detected_traced", "prop_contacts_with_detected_index"],
    )
