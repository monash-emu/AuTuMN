class AbsPropIsolatedProc:
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """
    def __init__(self, age_idx, abs_props, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.early_rate = early_rate

    def __call__(self, time, input_values):
        return get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, input_values["cdr"]) * self.early_rate

class AbsPropSymptNonHospProc:
    """
    Returns the absolute proportion of infected not entering the hospital.
    This also does not count people who are isolated/detected.
    """
    def __init__(self, age_idx, abs_props, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.early_rate = early_rate

    def get_abs_prop_sympt_non_hospital(self, time, cdr):
        prop_isolated = get_abs_prop_isolated(self.proportion_sympt, self.proportion_hosp, cdr)
        return self.proportion_sympt - self.proportion_hosp - prop_isolated

    def __call__(self, time, input_values):
        return self.get_abs_prop_sympt_non_hospital(time, input_values["cdr"]) * self.early_rate


def get_abs_prop_isolated(proportion_sympt, proportion_hosp, cdr):
    target_prop_detected = proportion_sympt * cdr
    return max(0.0, target_prop_detected - proportion_hosp)
