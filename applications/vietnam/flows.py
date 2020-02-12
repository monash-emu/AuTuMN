from autumn.tb_model.flows import (
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
)
from summer_py.summer_model import StratifiedModel, find_name_components, find_stem
from summer_py.constants import Compartment, Flow

from .rate_builder import RateBuilder


def get_flows(rates: RateBuilder, strain_params: dict):
    """
    Returns a list of compartmental model flows.
    """
    flows = []
    flows = add_standard_infection_flows(flows)
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)
    return [
        *flows,
        # Add case detection process to basic model
        {
            "type": Flow.STANDARD,
            "parameter": "case_detection",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
        # Add Isoniazid preventive therapy flow
        {
            "type": Flow.CUSTOM,
            "parameter": "ipt_rate",
            "origin": Compartment.EARLY_LATENT,
            "to": Compartment.RECOVERED,
            "function": build_ipt_flow_func(rates, strain_params),
        },
        # Add active case finding flow
        {
            "type": Flow.STANDARD,
            "parameter": "acf_rate",
            "origin": Compartment.INFECTIOUS,
            "to": Compartment.RECOVERED,
        },
    ]


def build_ipt_flow_func(rates: RateBuilder, strain_params: dict):
    def ipt_flow_func(model: StratifiedModel, n_flow: int, time: int, compartment_values):
        """
        Work out the number of detected individuals from the relevant active TB compartments (with regard to the origin
        latent compartment of n_flow) multiplied with the proportion of the relevant infected contacts that is from this
        latent compartment.
        """
        dict_flows = model.transition_flows_dict
        origin_comp_name = dict_flows["origin"][n_flow]
        components_latent_comp = find_name_components(origin_comp_name)

        # Find compulsory tags to be found in relevant infectious compartments
        tags = []
        for component in components_latent_comp:
            if "location_" in component or "strain_" in component:
                tags.append(component)

        # loop through all relevant infectious compartments
        total_tb_detected = 0.0
        for comp_ind in model.infectious_indices["all_strains"]:
            active_components = find_name_components(model.compartment_names[comp_ind])
            if all(elem in active_components for elem in tags):
                infectious_pop = compartment_values[comp_ind]
                detection_indices = [
                    index
                    for index, val in dict_flows["parameter"].items()
                    if "case_detection" in val
                ]
                flow_index = [
                    index
                    for index in detection_indices
                    if dict_flows["origin"][index] == model.compartment_names[comp_ind]
                ][0]
                param_name = dict_flows["parameter"][flow_index]
                detection_tx_rate = model.get_parameter_value(param_name, time)
                tsr = rates.get_treatment_success(time)
                if "strain_mdr" in model.compartment_names[comp_ind]:
                    tsr = strain_params["mdr_tsr"] * strain_params["prop_mdr_detected_as_mdr"]
                if tsr > 0.0:
                    total_tb_detected += infectious_pop * detection_tx_rate / tsr

        # list all latent compartments relevant to the relevant infectious population
        relevant_latent_compartments_indices = [
            i
            for i, comp_name in enumerate(model.compartment_names)
            if find_stem(comp_name) == "early_latent" and all(elem in comp_name for elem in tags)
        ]

        total_relevant_latent_size = sum(
            compartment_values[i] for i in relevant_latent_compartments_indices
        )
        current_latent_size = compartment_values[model.compartment_names.index(origin_comp_name)]
        prop_of_relevant_latent = (
            current_latent_size / total_relevant_latent_size
            if total_relevant_latent_size > 0.0
            else 0.0
        )

        return total_tb_detected * prop_of_relevant_latent

    return ipt_flow_func
