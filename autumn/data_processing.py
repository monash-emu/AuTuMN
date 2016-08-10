
import autumn.spreadsheet as spreadsheet
import copy
import numpy
import warnings
import tool_kit

class Inputs:

    def __init__(self, from_test=False):

        self.from_test = from_test

    def read_and_load_data(self, keys_of_sheets_to_read=('bcg', 'rate_birth', 'life_expectancy', 'control_panel',
                                                         'default_parameters', 'tb', 'notifications', 'outcomes',
                                                         'country_constants', 'default_constants', 'country_economics',
                                                         'default_economics', 'country_programs', 'default_programs',
                                                         'diabetes')):

        # Determine country to be read
        self.determine_country()

        # Read the relevant sheets
        self.read_sheets(keys_of_sheets_to_read)

        # Work through the data processing
        self.derived_data = {}
        self.find_organ_proportions()
        self.time_variants = {}
        self.find_time_variants()
        self.update_time_variants()
        self.add_timevariant_defaults()
        self.model_constants = {}
        self.add_model_constant_defaults()
        self.find_ds_outcomes()
        self.add_read_treatment_outcomes()
        if self.model_constants['n_strains'] > 1:
            self.duplicate_ds_outcomes_for_multistrain()
        self.add_epi_dictionaries_to_timevariants()
        self.add_organ_status_to_timevariants()
        self.add_resistant_strain_outcomes()
        self.tidy_timevariants()
        self.add_economic_timevariants()
        self.add_fixed_parameters()
        self.find_age_groups()
        if len(self.agegroups) > 1:
            self.set_fixed_age_specific_parameters()
            self.find_ageing_rates()

        # Perform checks
        self.checks()

    def determine_country(self):

        """
        Method to quickly return the country for analysis by reading the control panel
        and only extracting the country name.
        """

        self.country = spreadsheet.read_input_data_xls(self.from_test,
                                                       ['control_panel'])['control_panel']['country']

    def read_sheets(self, keys_of_sheets_to_read):

        """
        Method that just calls the reading function in the spreadsheet reader,
        with directory correction and country specification as required.

        Args:
            keys_of_sheets_to_read: Specifies the spreadsheets that need to be read,
                defaults are provided through read_and_load_data method inputs above.
        """

        self.original_data \
            = spreadsheet.read_input_data_xls(self.from_test,
                                              keys_of_sheets_to_read,
                                              self.country)

    def find_time_variants(self):

        """
        Extracts all the country-specific economic variables and populates the time-variants
        dictionary with them.
        """

        if 'country_programs' in self.original_data:
            self.time_variants.update(self.original_data['country_programs'])

    def update_time_variants(self):

        """
        Adds vaccination and case detection time-variants to the manually entered data
        loaded from the spreadsheets.
        Note that the manual inputs always over-ride the loaded data if both are present.
        """

        # Vaccination
        if self.time_variants['program_prop_vaccination']['load_data'] == 'yes':
            for year in self.original_data['bcg']:
                # If not already loaded through the inputs spreadsheet
                if year not in self.time_variants['program_prop_vaccination']:
                    self.time_variants['program_prop_vaccination'][year] \
                        = self.original_data['bcg'][year]

        # Case detection
        if self.time_variants['program_prop_detect']['load_data'] == 'yes':
            for year in self.original_data['tb']['c_cdr']:
                # If not already loaded through the inputs spreadsheet
                if year not in self.time_variants['program_prop_detect']:
                    self.time_variants['program_prop_detect'][year] \
                        = self.original_data['tb']['c_cdr'][year]

    def find_organ_proportions(self):

        """
        Calculates dictionaries with proportion of cases progressing to each organ status by year,
        and adds these to the derived_data attribute of the object.
        """

        self.derived_data.update(self.calculate_proportion_dict(self.original_data['notifications'],
                                                                ['new_sp', 'new_sn', 'new_ep']))

    def add_timevariant_defaults(self):

        """
        Populates time variant parameters with defaults if those values aren't found
        in the manually entered country-specific data.
        """

        for program_var in self.original_data['default_programs']:

            # If the key isn't in available for the country at all
            if program_var not in self.time_variants:
                self.time_variants[program_var] = \
                    self.original_data['default_programs'][program_var]

            # Otherwise if it's there, populate for the missing years
            else:
                for year in self.original_data['default_programs'][program_var]:
                    if year not in self.time_variants[program_var]:
                        self.time_variants[program_var][year] = \
                            self.original_data['default_programs'][program_var][year]

    def add_model_constant_defaults(self,
                                    other_sheets_with_constants=('diabetes', 'country_constants', 'default_constants')):

        """
        Populate model_constants with data from control panel, country sheet or default sheet hierarhically
        - such that the control panel is read in preference to the country data in preference to the default back-ups
        Args:
            other_sheets_with_constants: The sheets of original_data which contain model constants
        """

        self.model_constants = self.original_data['control_panel']
        for other_sheet in other_sheets_with_constants:
            if other_sheet in self.original_data:
                for item in self.original_data[other_sheet]:
                    if item not in self.original_data['control_panel']:
                        self.model_constants[item] = \
                            self.original_data[other_sheet][item]

    def find_ds_outcomes(self):

        """
        Calculates proportions of patients with each reported outcome for DS-TB,
        then sums cure and completion to obtain treatment success proportion.
        Note that the outcomes are reported differently for resistant strains, so code differs for them.
        """

        # Calculate each proportion
        self.derived_data.update(
            self.calculate_proportion_dict(
                self.original_data['outcomes'],
                ['new_sp_cmplt', 'new_sp_cur', 'new_sp_def', 'new_sp_died', 'new_sp_fail'],
                percent=True))

        # Sum to get treatment success
        self.derived_data['prop_new_sp_success'] = {}
        for year in self.derived_data['prop_new_sp_cmplt']:
            self.derived_data['prop_new_sp_success'][year] \
                = self.derived_data['prop_new_sp_cmplt'][year] \
                  + self.derived_data['prop_new_sp_cur'][year]

    def add_read_treatment_outcomes(self):

        """
        Add treatment outcomes for DS-TB to the time variants attribute.
        Use the same approach as above to adding if requested and if data not manually entered.
        """

        # Iterate over success and death outcomes
        for outcome in ['_success', '_death']:

            # Correct naming GTB report
            if outcome == '_success':
                report_outcome = '_success'
            elif outcome == '_death':
                report_outcome = '_died'

            # Populate data
            if self.time_variants['program_prop_treatment' + outcome]['load_data'] == 'yes':
                for year in self.derived_data['prop_new_sp' + report_outcome]:
                    if year not in self.time_variants['program_prop_treatment' + outcome]:
                        self.time_variants['program_prop_treatment' + outcome][year] \
                            = self.derived_data['prop_new_sp' + report_outcome][year]

    def duplicate_ds_outcomes_for_multistrain(self):

        """
        Duplicates the treatment outcomes with DS-TB key if it is a multi-strain model.
        """

        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_ds'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome])

    def add_epi_dictionaries_to_timevariants(self):

        """
        Add epidemiological time variant parameters to time_variants.
        Similarly to previous methods, only performed if requested
        and only populated where absent (either entirely or for that year).
        """

        for demo_parameter in ['life_expectancy', 'rate_birth']:
            if self.time_variants['demo_' + demo_parameter]['load_data'] == 'yes':
                for year in self.original_data[demo_parameter]:
                    if year not in self.time_variants['demo_' + demo_parameter]:
                        self.time_variants['demo_' + demo_parameter][year] \
                            = self.original_data[demo_parameter][year]

    def add_organ_status_to_timevariants(self):

        """
        Populate organ status dictionaries where requested and not already loaded.
        """

        if self.time_variants['epi_prop_smearpos']['load_data'] == 'yes':

            # Iterate over smear positive and negative status
            for outcome in ['_smearpos', '_smearneg']:

                # Correct naming GTB report
                if outcome == '_smearpos':
                    report_outcome = '_sp'
                elif outcome == '_smearneg':
                    report_outcome = '_sn'

                # Populate data
                for year in self.derived_data['prop_new' + report_outcome]:
                    if year not in self.time_variants['epi_prop' + outcome]:
                        self.time_variants['epi_prop' + outcome][year] \
                            = self.derived_data['prop_new' + report_outcome][year]

    def add_resistant_strain_outcomes(self):

        """
        Finds treatment outcomes for the resistant strains (i.e. MDR and XDR-TB).
        As for DS-TB, no need to find default proportion, as it is equal to one minus success minus death.
        **** Inappropriate outcomes are currently set to those for XDR-TB - this is temporary ****
        """

        # Calculate proportions of each outcome for MDR and XDR-TB from GTB
        for strain in ['mdr', 'xdr']:
            self.derived_data.update(
                self.calculate_proportion_dict(self.original_data['outcomes'],
                                               [strain + '_succ', strain + '_fail', strain + '_died', strain + '_lost'],
                                               percent=True))

            # Populate MDR and XDR data from outcomes dictionary into time variants where requested and not entered
            if self.time_variants['program_prop_treatment_success_' + strain]['load_data'] == 'yes':
                for year in self.derived_data['prop_' + strain + '_succ']:
                    if year not in self.time_variants['program_prop_treatment_success_' + strain]:
                        self.time_variants['program_prop_treatment_success_' + strain][year] \
                            = self.derived_data['prop_' + strain + '_succ'][year]

        # Temporarily assign the same treatment outcomes to XDR-TB as for inappropriate
        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_inappropriate'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome + '_xdr'])

    def tidy_timevariants(self):

        """
        Perform final rounds of tidying of time-variants
        """

        # Looping over each time variant
        for program in self.time_variants:

            # Add zero at starting time for model run to all programs that are proportions
            if 'program_prop' in program:
                self.time_variants[program][int(self.model_constants['start_time'])] = 0.

            # Remove the load_data keys, as they have been used and are now redundant
            self.time_variants[program] \
                = self.remove_specific_key(self.time_variants[program], 'load_data')

            # Remove dictionary keys for which values are nan
            self.time_variants[program] \
                = self.remove_nans(self.time_variants[program])

    def add_economic_timevariants(self):

        """
        Method that probably needs some work and Tan should feel free to change.
        Currently loads default and country-specific economic variables, with the country-specific ones
        over-riding the defaults where present (for consistency with epidemiological approach).
        """

        # Add all the loaded country-specific economic variables to the time-variants dictionary
        if 'country_economics' in self.original_data:
            self.time_variants.update(self.original_data['country_economics'])

        # Add default values if country_specific ones not available
        for economic_var in self.original_data['default_economics']:
            if economic_var not in self.time_variants:
                self.time_variants[economic_var] \
                    = self.original_data['default_economics'][economic_var]

    def add_fixed_parameters(self):

        """
        Sets parameters that should never be changed in any situation,
        i.e. "by definition" parameters (although note that the infectiousness
        of the single infectious compartment for models unstratified by organ
        status is now set in set_fixed_infectious_proportion, because it is
        dependent upon loading some parameters in find_functions_or_params)
        """

        if self.model_constants['n_organs'] < 2:
            # Proportion progressing to the only infectious compartment
            # for models unstratified by organ status
            self.model_constants['epi_prop'] = 1.
        else:
            self.model_constants['tb_multiplier_force_smearpos'] \
                = 1.  # Infectiousness of smear-positive patients
            self.model_constants['tb_multiplier_force_extrapul'] \
                = 0.  # Infectiousness of extrapulmonary patients

    def find_age_groups(self):

        # Age stratification
        self.agegroups, _ = \
            tool_kit.get_agegroups_from_breakpoints(
                self.model_constants['age_breakpoints'])

    def set_fixed_age_specific_parameters(self):

        # Extract age breakpoints in appropriate form for module
        model_breakpoints = []
        for i in self.model_constants['age_breakpoints']:
            model_breakpoints += [float(i)]

        for param in ['early_progression_age', 'late_progression_age',
                      'tb_multiplier_child_infectiousness_age']:
            # Extract age-stratified parameters in the appropriate form
            prog_param_vals = {}
            prog_age_dict = {}
            for constant in self.model_constants:
                if param in constant:
                    prog_param_string, prog_stem = \
                        tool_kit.find_string_from_starting_letters(constant, '_age')
                    prog_age_dict[prog_param_string], _ = \
                        tool_kit.interrogate_age_string(prog_param_string)
                    prog_param_vals[prog_param_string] = \
                        self.model_constants[constant]

            param_breakpoints = tool_kit.find_age_breakpoints_from_dicts(prog_age_dict)

            # Find and set age-adjusted parameters
            prog_age_adjusted_params = \
                tool_kit.adapt_params_to_stratification(param_breakpoints,
                                                        model_breakpoints,
                                                        prog_param_vals,
                                                        parameter_name=param)
            for agegroup in self.agegroups:
                self.model_constants[prog_stem + agegroup] = prog_age_adjusted_params[agegroup]

    def find_ageing_rates(self):

        """
        Calculate ageing rates as the reciprocal of the width of the age bracket.
        """

        for agegroup in self.agegroups:
            age_limits, _ = tool_kit.interrogate_age_string(agegroup)
            if 'up' not in agegroup:
                self.model_constants['ageing_rate' + agegroup] \
                    = 1. / (age_limits[1] - age_limits[0])

    def checks(self):

        """
        Perform checks for data consistency
        """

        for status in ['pos', 'neg']:

            # If no organ stratification is requested, but time variant organ status requested
            if self.model_constants['n_organs'] < 2 \
                    and self.time_variants['epi_prop_smear' + status]['time_variant'] == 'yes':

                # Warn about the problem
                warnings.warn('Warning: time variant smear-' + status + ' proportion requested, but ' +
                              'model is not stratified by organ status. Therefore, time variant smear-' + status +
                              ' status has been turned off.')

                # Make the proportions constant instead
                self.time_variants['epi_prop_smear' + status]['time_variant'] = 'no'

    #############################################################################
    #  General methods for use by the other methods above

    def calculate_proportion_dict(self, data, indices, percent=False):

        """
        General method to calculate proportions from absolute values provided as dictionaries.

        Args:
            data: Dicionary containing the absolute values.
            indices: The keys of data from which proportions are to be calculated (generally a list of strings).
            percent: Boolean describing whether the method should return the output as a percent or proportion.

        Returns:
            proportions: A dictionary of the resulting proportions.
        """

        # Calculate multiplier for percentages if requested, otherwise leave as one
        if percent:
            multiplier = 1E2
        else:
            multiplier = 1.

        # Create a list of the years that are common to all indices within data
        lists_of_years = []
        for i in range(len(indices)):
            lists_of_years += [data[indices[i]].keys()]
        common_years = self.find_common_elements_multiple_lists(lists_of_years)

        # Calculate the denominator by summing the values for which proportions have been requested
        denominator = {}
        for i in common_years:
            for j in indices:
                if j == indices[0]:
                    denominator[i] = data[j][i]
                else:
                    denominator[i] += data[j][i]

        # Calculate the proportions
        proportions = {}
        for j in indices:
            proportions['prop_' + j] = {}
            for i in common_years:
                if denominator[i] > 0.:
                    proportions['prop_' + j][i] = \
                        data[j][i] / denominator[i] \
                        * multiplier

        return proportions

    def find_common_elements_multiple_lists(self, list_of_lists):

        """
        Simple method to find the common elements of any number of lists

        Args:
            list_of_lists: A list whose elements are all the lists we want to find the
                intersection of.

        Returns:
            intersection: Common elements of all lists.
        """

        intersection = list_of_lists[0]
        for i in range(1, len(list_of_lists)):
            intersection = self.find_common_elements(intersection, list_of_lists[i])
        return intersection

    def find_common_elements(self, list_1, list_2):

        """
        Simple method to find the intersection of two lists

        Args:
            list_1 and list_2: The two lists

        Returns:
            intersection: The common elements of the two lists
        """

        intersection = []
        for i in list_1:
            if i in list_2:
                intersection += [i]
        return intersection

    def remove_specific_key(self, dictionary, key):

        """
        Remove a specific named key from a dictionary

        Args:
            dictionary: The dictionary to have a key removed
            key: The key to be removed

        Returns:
            dictionary: The dictionary with the key removed
        """

        if key in dictionary:
            del dictionary[key]

        return dictionary

    def remove_nans(self, dictionary):

        """
        Takes a dictionary and removes all of the elements for which the value is nan

        Args:
            dictionary: Should typically be the dictionary of programmatic values, usually
                        with time in years as the key.

        Returns:
            dictionary: The dictionary with the nans removed.
        """

        nan_indices = []
        for i in dictionary:
            if type(dictionary[i]) == float and numpy.isnan(dictionary[i]):
                nan_indices += [i]
        for i in nan_indices:
            del dictionary[i]

        return dictionary


if __name__ == '__main__':

    inputs = Inputs()
    inputs.read_and_load_data()
    print()


