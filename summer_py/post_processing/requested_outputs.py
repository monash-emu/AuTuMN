from summer_py.summer_model.utils.string import find_name_components


class RequestedOutput:
    """
    Base post-processing output class
    """

    PREVLAENCE = "prev"
    STRATA_DISTRIBUTION = "distribution_of_strata"
    TYPES = (PREVLAENCE, STRATA_DISTRIBUTION)

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def from_str(cls, s):
        if type(s) in [PrevalenceOutput, StrataOutput]:
            return s
        elif s.startswith(cls.PREVLAENCE):
            return PrevalenceOutput.from_str(s)
        elif s.startswith(cls.STRATA_DISTRIBUTION):
            return StrataOutput.from_str(s)
        else:
            raise ValueError(f"Unrecognised requested output f{s}")

    def to_str(self):
        raise NotImplementedError()

    def is_valid(self, model):
        raise NotImplementedError()

    def calculate_output(self, model, multipliers, time_indices):
        raise NotImplementedError()


class PrevalenceOutput(RequestedOutput):
    """
    Post processing of "prevalence".
    """

    type = RequestedOutput.PREVLAENCE

    def __init__(self, numerator_conditions, denominator_conditions):
        """
        numerator_conditions: array of strata names, eg. ["infectious"]
        denominator_conditions: dict of strata, eg.  {"age": ["age_0", "age_5"], "bcg": ["bcg_vaccinated"]}
        """
        self.numerator_conditions = numerator_conditions
        self.denominator_conditions = denominator_conditions

    @classmethod
    def from_str(cls, s):
        pre_among, post_among = s.split("among")
        numerator_conditions = pre_among.split("X")[1:-1]
        denominator_categories = post_among.split("X")[1:]
        denominator_conditions = {}
        for category in denominator_categories:
            stratification = category.split("_")[0]
            if stratification not in denominator_conditions.keys():
                denominator_conditions[stratification] = []

            denominator_conditions[stratification].append(category)

        return PrevalenceOutput(numerator_conditions, denominator_conditions)

    def to_str(self):
        denominator_categories = []
        for conditions in self.denominator_conditions.values():
            denominator_categories += conditions

        items = [self.PREVLAENCE] + self.numerator_conditions + ["among"] + denominator_categories
        return "X".join(items)

    def is_valid(self, model):
        return True

    def calculate_output(self, model, multipliers, time_indices):
        """
        Returns the requested output for a given list of time indices.
        """
        numerator_indices = []
        denominator_indices = []
        for comp_idx, compartment in enumerate(model.compartment_names):
            name_components = find_name_components(compartment)
            is_in_denominator = True
            for condition in self.denominator_conditions.keys():
                if not any(
                    category in name_components
                    for category in self.denominator_conditions[condition]
                ):
                    is_in_denominator = False
                    break

            if is_in_denominator:
                if all(category in compartment for category in self.numerator_conditions):
                    numerator_indices.append(comp_idx)
                else:
                    denominator_indices.append(comp_idx)

        generated_output = []
        for time_idx in time_indices:
            numerator = model.outputs[time_idx, numerator_indices].sum()
            extra_for_denominator = model.outputs[time_idx, denominator_indices].sum()
            try:
                value = numerator / (numerator + extra_for_denominator)
            except ZeroDivisionError:
                value = 0

            output_str = self.to_str()
            if output_str in multipliers.keys():
                value *= multipliers[output_str]

            generated_output.append(value)

        return generated_output


class StrataOutput(RequestedOutput):
    """
    Post processing of "distribution of strata".
    """

    type = RequestedOutput.STRATA_DISTRIBUTION

    def __init__(self, stratum):
        self.stratum = stratum

    @classmethod
    def from_str(cls, s):
        stratum = s.split("X")[1]
        return StrataOutput(stratum)

    def to_str(self):
        items = [self.STRATA_DISTRIBUTION, self.stratum]
        return "X".join(items)

    def is_valid(self, model):
        valid_strats = model.all_stratifications.keys()
        if self.stratum not in valid_strats:
            print(f"Warning: {self.stratum} is not a model stratification. Will be ignored.")
            return False
        else:
            return True

    def calculate_output(self, model, multipliers, time_indices):
        compartment_indices = {}
        # Populate with the indices of the compartment of interest
        for stratum_name in model.all_stratifications[self.stratum]:
            compartment_indices[stratum_name] = []
            for i_comp, compartment_name in enumerate(model.compartment_names):
                name_components = find_name_components(compartment_name)
                if self.stratum + "_" + stratum_name in name_components:
                    compartment_indices[stratum_name].append(i_comp)

        out = {}
        for stratum in compartment_indices.keys():
            out[stratum] = []
            for i_time in time_indices:
                value = model.outputs[i_time, compartment_indices[stratum]].sum()
                out[stratum].append(value)

        return out
