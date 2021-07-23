class AbsPropIsolatedProc:
    """
    Returns the absolute proportion of infected becoming isolated at home.
    Isolated people are those who are detected but not sent to hospital.
    """
    def __init__(self, age_idx, abs_props, get_detected_proportion, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.get_detected_proportion = get_detected_proportion
        self.early_rate = early_rate

    def get_abs_prop_isolated(self, time):
        target_prop_detected = self.proportion_sympt * self.get_detected_proportion(time)

        return max(0.0, target_prop_detected - self.proportion_hosp)

    def __call__(self, time):
        return self.get_abs_prop_isolated(time) * self.early_rate

class AbsPropSymptNonHospProc:
    """
    Returns the absolute proportion of infected not entering the hospital.
    This also does not count people who are isolated/detected.
    """
    def __init__(self, age_idx, abs_props, get_detected_proportion, early_rate):
        self.age_idx = age_idx
        self.proportion_sympt = abs_props["sympt"][age_idx]
        self.proportion_hosp = abs_props["hospital"][age_idx]
        self.get_detected_proportion = get_detected_proportion
        self.early_rate = early_rate

    def get_abs_prop_isolated(self, time):
        target_prop_detected = self.proportion_sympt * self.get_detected_proportion(time)

        return max(0.0, target_prop_detected - self.proportion_hosp)

    def get_abs_prop_sympt_non_hospital(self, time):
        return self.proportion_sympt - self.proportion_hosp - self.get_abs_prop_isolated(time)

    def __call__(self, time):
        return self.get_abs_prop_sympt_non_hospital(time) * self.early_rate

