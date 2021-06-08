from autumn.models.covid_19.constants import Compartment


def trace_function(prop_traced, incidence_flow_rate, untraced_comp):
    def contact_tracing_func(
            model, compartments, compartment_values, flows, flow_rates, time, source=untraced_comp
    ):

        # Currently unused, but proof of concept - unfortunately adds ridiculously to run-time
        active_comps = \
            [idx for idx, comp in enumerate(compartments) if
             comp.has_name(Compartment.EARLY_ACTIVE) or comp.has_name(Compartment.LATE_ACTIVE)]
        prevalence = sum(compartment_values[active_comps]) / sum(compartment_values)

        return incidence_flow_rate * prop_traced * compartment_values[source.idx]

    return contact_tracing_func
