from autumn.curve import scale_up_function
from autumn.tb_model import convert_competing_proportion_to_rate
from autumn.tool_kit import return_function_of_function


class RateBuilder:
    """
    Builds functions that describe time-varying model parameters.
    """

    def __init__(self, rate_params, tb_control_recovery_rate_organ):
        self.rate_params = rate_params
        self.tb_control_recovery_rate_organ = tb_control_recovery_rate_organ
        cdr = rate_params["case_detection_rate"]
        tsr = rate_params["treatment_success_rate"]
        self.cdr_func = scale_up_function(
            cdr.keys(),
            [c * rate_params["case_detection_rate_multiplier"] for c in list(cdr.values())],
            smoothness=0.2,
            method=5,
        )
        self.tsr_func = scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)
        cdr_by_organ = {
            "smearpos": self.get_case_detection_smearpos,
            "smearneg": self.get_case_detection_smearneg,
            "extrapul": self.get_case_detection_extrapul,
            "overall": self.get_case_detection,
        }
        disease_duration = rate_params["disease_duration"]
        self.detect_rate_by_organ = {}
        for organ in cdr_by_organ.keys():
            prop_to_rate = convert_competing_proportion_to_rate(1.0 / disease_duration[organ])
            self.detect_rate_by_organ[organ] = return_function_of_function(
                cdr_by_organ[organ], prop_to_rate
            )

    def get_tb_control_recovery(self, time):
        """
        Get Tuberculosis control recovery rate for a given timestep.
        """
        get_detect_rate = self.detect_rate_by_organ[self.tb_control_recovery_rate_organ]
        return self.get_treatment_success(time) * get_detect_rate(time)

    def get_isoniazid_preventative_therapy(self, time):
        """
        Get Isoniazid preventive therapy (IPT) at a given timestep.
        Assume a coverage of 1.0 before age stratification.
        """
        ipt_params = self.rate_params["ipt"]
        return ipt_params["yield_contact_ct_tstpos_per_detected_tb"] * ipt_params["ipt_efficacy"]

    def get_active_case_finding(self, time):
        """
        Returns that active case finding rate at a given timestep.
        """
        acf_params = self.rate_params["acf"]
        return acf_params["coverage"] * acf_params["sensitivity"] * self.get_treatment_success(time)

    def get_treatment_success(self, time):
        """
        Returms the treatment success rate at a given timestep.
        """
        tsr = self.tsr_func(time)
        return tsr + self.rate_params["reduction_negative_tx_outcome"] * (1.0 - tsr)

    def get_case_detection(self, time):
        """
        Returns the case detection rate at a given timestep.
        """
        return self.cdr_func(time)

    def get_case_detection_smearpos(self, time):
        """
        Work out the case detection rate for smear-positive TB.
        """
        props = self.rate_params["organ_proportions"]
        sens = self.rate_params["diagnostic_sensitivity"]
        return self.get_case_detection(time) / (
            props["smearpos"]
            + props["smearneg"] * sens["smearneg"]
            + props["extrapul"] * sens["extrapul"]
        )

    def get_case_detection_smearneg(self, time):
        """
        Work out the case detection rate for smear-negative TB.
        """
        sens = self.rate_params["diagnostic_sensitivity"]
        return self.get_case_detection_smearpos(time) * sens["smearneg"]

    def get_case_detection_extrapul(self, time):
        """
        Work out the case detection rate for extrapul TB.
        """
        sens = self.rate_params["diagnostic_sensitivity"]
        return self.get_case_detection_smearpos(time) * sens["extrapul"]

    def get_dr_amplification(self, time):
        """
        Returns the "dr_amplification_rate" for a given timestep.
        """
        get_detect_rate = self.detect_rate_by_organ["overall"]
        return (
            get_detect_rate(time)
            * (1.0 - self.tsr_func(time))
            * (1.0 - self.rate_params["reduction_negative_tx_outcome"])
            * self.rate_params["dr_amplification_prop_among_nonsuccess"]
        )
