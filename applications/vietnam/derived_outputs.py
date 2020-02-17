"""
Create some customised derived_outputs
"""
from summer_py.summer_model import StratifiedModel

from .rate_builder import RateBuilder


def build_calc_notifications(stratum: str, rates: RateBuilder, strain_params: dict):
    """
    Build a notification function.
    Example of stratum: "Xage_0Xstrain_mdr"
    """

    def calc_notifications(model: StratifiedModel, time: int):
        """
        Not sure what this does.
        """
        total_notifications = 0.0
        dict_flows = model.transition_flows_dict
        compartnment_name = f"infectious{stratum}"
        comp_idx = model.compartment_idx_lookup[compartnment_name]
        infectious_pop = model.compartment_values[comp_idx]
        detection_indices = [
            index for index, val in dict_flows["parameter"].items() if "case_detection" in val
        ]
        flow_index = [
            index
            for index in detection_indices
            if dict_flows["origin"][index] == model.compartment_names[comp_idx]
        ][0]
        param_name = dict_flows["parameter"][flow_index]
        detection_tx_rate = model.get_parameter_value(param_name, time)
        tsr = rates.get_treatment_success(time)
        if "strain_mdr" in model.compartment_names[comp_idx]:
            tsr = strain_params["mdr_tsr"] * strain_params["prop_mdr_detected_as_mdr"]
        if tsr > 0.0:
            total_notifications += infectious_pop * detection_tx_rate / tsr

        return total_notifications

    return calc_notifications


def build_calc_num_detected(tag: str):
    """
    example of tag: "starin_mdr" or "organ_smearpos"
    """

    def calc_num_detected(model: StratifiedModel, time: int):
        """
        Not sure what this does.
        """
        nb_treated = 0.0
        for key, value in model.derived_outputs.items():
            if "notifications" in key and tag in key:
                this_time_index = model.times.index(time)
                nb_treated += value[this_time_index]
        return nb_treated

    return calc_num_detected


def build_calc_popsize_acf(rate_params: dict):
    def calc_popsize_acf(model: StratifiedModel, time: int):
        """
        Not sure what this does.
        active case finding popsize: number of people screened
        """
        if rate_params["acf"]["coverage"] == 0.0:
            return 0.0

        pop_urban_ger = sum(
            [
                model.compartment_values[i]
                for i, c_name in enumerate(model.compartment_names)
                if "location_urban_ger" in c_name
            ]
        )
        return rate_params["acf"]["coverage"] * pop_urban_ger

    return calc_popsize_acf
