from numpy import linspace

from summer.model.strat_model import StratifiedModel
from summer.model.utils.string import find_name_components
from summer.model.utils.data_structures import *


class PostProcessing:
    """
    This class handles the calculations to be made after model integration in order to produce the requested outputs.

    :attribute model: an EpiModel object. This is the model from which the outputs need to be interpreted. The mode must
     have been run.
    :attribute requested_outputs: list of string variables defining the outputs to be calculated. This string will be
    interpreted automatically.
    :attribute requested_times: dictionary with keys that have to be listed in requested_outputs and values that are the
    associated lists of time points where the outputs are requested. If an output listed in requested_outputs is not
    one of the keys of requested_times, its values will be computed for all the model time points.
    :attribute operations_to_perform: dictionary containing the operations to be made in order to produce the requested
    outputs. The keys are the strings listed in requested_outputs. See interpret_requested_outputs for definition of
    the values.
    :attribute generated_outputs: dictionary storing the newly generated outputs.
    """

    def __init__(
        self,
        model,
        requested_outputs,
        scenario_number=0,
        requested_times={},
        multipliers={},
        ymax={},
    ):
        self.model = model
        self.requested_outputs = requested_outputs
        self.scenario_number = scenario_number
        self.requested_times = requested_times
        self.multipliers = multipliers
        self.ymax = ymax
        self.derived_outputs = model.derived_outputs if hasattr(model, "derived_outputs") else {}

        self.operations_to_perform = {}
        self.generated_outputs = {}

        self.check()
        self.interpret_requested_outputs()
        self.generate_outputs()

    def check(self):
        """
        Perform a few checks on the PostProcessing object's attributes.
        """
        # Check that the model has been run
        if self.model.outputs is None:
            raise ValueError("the model needs to be run before post-processing")

        # Check that the keys of self.requested_times are all listed in self.requested_outputs
        if not set(self.requested_times.keys()).issubset(self.requested_outputs):
            raise ValueError("all keys of 'requested_times' must be in 'requested_outputs'")

        # Check that the requested strata distributions correspond to an existing model stratification
        indices_to_be_removed = []
        for i_output, output in enumerate(self.requested_outputs):
            if output.startswith("distribution_of_strata"):
                stratification_of_interest = output.split("X")[1]
                if stratification_of_interest not in self.model.all_stratifications.keys():
                    print(
                        "Warning: Requested stratification '"
                        + stratification_of_interest
                        + "' is not among the model stratifications. Will be ignored for output processing"
                    )
                    indices_to_be_removed.append(i_output)

        self.requested_outputs = remove_multiple_elements_from_list(
            self.requested_outputs, indices_to_be_removed
        )

    def interpret_requested_outputs(self):
        """
        Interpret the string variables provided in requested outputs to define the calculations to be made. This method
        will populate self.operations_to_perform.

        The requested output should be in the following format: 'prevXlatentXstrain_mdrXamongXage_0Xage_5Xbcg_vaccinated'
        - keywords are separated with the character 'X'
        - the first keyword indicates the type of measure (prev, inc, ...)
        - the keywords located after 'among' specify the population of interest for an output:
          A compartment will be considered relevant if its name contains any one of the strata strings for each of
          the stratification factors listed. With the example above, a compartment's name needs to contain
          ('age_0' OR 'age_5') AND 'bcg_vaccinated' to be considered.
        - the keywords located before 'among' (excluding the first keyword) specify the infection states relevant to the
          output of interest. With the example above, we are interested in individuals who have latent infection with a
          MDR strain.
        """
        for output in self.requested_outputs:
            self.operations_to_perform[output] = {}
            if output.startswith("prev"):
                string_pre_among, string_post_among = output.split("among")

                # collate all the stratification conditions to be satisfied
                denominator_categories = string_post_among.split("X")[1:]
                # e.g. ['age_0', 'age_5', 'bcg_vaccinated']
                denominator_conditions = {}
                for category in denominator_categories:
                    stratification = category.split("_")[0]
                    if stratification not in denominator_conditions.keys():
                        denominator_conditions[stratification] = []
                    denominator_conditions[stratification].append(category)
                # e.g. {'age': ['age_0', 'age_5'], 'bcg': ['bcg_vaccinated']}

                # work out the conditions to be satisfied regarding the compartment of interest
                numerator_conditions = string_pre_among.split("X")[1:-1]

                # list all relevant compartments that should be included into the numerator or the denominator
                self.operations_to_perform[output]["numerator_indices"] = []

                # indices to be added to the numerator ones to form the whole denominator
                self.operations_to_perform[output]["denominator_extra_indices"] = []
                for i_comp, compartment in enumerate(self.model.compartment_names):
                    name_components = find_name_components(compartment)
                    is_in_denominator = True

                    # for each relevant stratification
                    for condition in denominator_conditions.keys():
                        if not any(
                            category in name_components
                            for category in denominator_conditions[condition]
                        ):
                            is_in_denominator = False
                            break

                    if is_in_denominator:
                        if all(category in compartment for category in numerator_conditions):
                            self.operations_to_perform[output]["numerator_indices"].append(i_comp)
                        else:
                            self.operations_to_perform[output]["denominator_extra_indices"].append(
                                i_comp
                            )

            # population distribution across a particular requested stratum
            elif output.startswith("distribution_of_strata"):

                # create dictionary keyed with the names of the strata within the stratification of interest
                self.operations_to_perform[output]["compartment_indices"] = {}
                stratification_of_interest = output.split("X")[1]

                # populate with the indices of the compartment of interest
                for stratum_name in self.model.all_stratifications[stratification_of_interest]:
                    self.operations_to_perform[output]["compartment_indices"][stratum_name] = []
                    for i_comp, compartment_name in enumerate(self.model.compartment_names):
                        if stratification_of_interest + "_" + stratum_name in find_name_components(
                            compartment_name
                        ):
                            self.operations_to_perform[output]["compartment_indices"][
                                stratum_name
                            ].append(i_comp)
            else:
                raise ValueError("only prevalence and distribution outputs are currently supported")

    def generate_outputs(self):
        """
        main method that generates all the requested outputs.
        'self.generated_outputs' is populated during this process
        """
        for output in self.requested_outputs:
            self.generated_outputs[output] = self.calculate_output_for_selected_times(
                output, self.find_output_times(output)
            )

    def find_output_times(self, output):
        """
        find the times for which outputs are required using the user-request if available
        """

        # find model output times for user-requested times if a request submitted
        if output in self.requested_times.keys():
            return [
                find_first_list_element_above(self.model.times, time)
                for time in self.requested_times[output]
            ]

        # otherwise calculate outputs for all model output times
        else:
            return range(len(self.model.times))

    def calculate_output_for_selected_times(self, output, time_indices):
        """
        returns the requested output for a given list of time indices

        :param output: the name of the requested output
        :param time_indices: the time index
        :return: the calculated value of the requested output at the requested time index
        """
        if output.startswith("prev"):
            out = []
            for i_time in time_indices:
                numerator = self.model.outputs[
                    i_time, self.operations_to_perform[output]["numerator_indices"]
                ].sum()
                extra_for_denominator = self.model.outputs[
                    i_time, self.operations_to_perform[output]["denominator_extra_indices"]
                ].sum()
                value = (
                    numerator / (numerator + extra_for_denominator)
                    if numerator + extra_for_denominator > 0.0
                    else 0
                )
                if output in self.multipliers.keys():
                    value *= self.multipliers[output]
                out.append(value)

        elif output.startswith("distribution_of_strata"):
            out = {}
            for stratum in self.operations_to_perform[output]["compartment_indices"].keys():
                out[stratum] = []
                for i_time in time_indices:
                    out[stratum].append(
                        self.model.outputs[
                            i_time,
                            self.operations_to_perform[output]["compartment_indices"][stratum],
                        ].sum()
                    )

        else:
            ValueError("output type not currently supported")
        return out

    def give_output_for_given_time(self, output, time):
        """
        quick method to return a specific output at a given time, once all requested outputs have been calculated.

        :param output: string
            specifies the output to be returned
        :param time: float
            time of interest
        :return:
            the requested output at the requested time
        """
        if output not in self.requested_outputs:
            raise ValueError("the output was not requested for calculation")

        elif output in self.requested_times.keys():
            if time not in self.requested_times[output]:
                raise ValueError("the time was not among the requested times for calculation")
            index = self.requested_times[output].index(time)

        else:
            if time < self.model.times[0] or time > self.model.times[-1]:
                raise ValueError("the requested time is not within the integration time range")
            index = find_first_list_element_above(self.model.times, time)

        return self.generated_outputs[output][index]
