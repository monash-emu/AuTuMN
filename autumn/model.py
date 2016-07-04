# -*- coding: utf-8 -*-


"""

The TB-specific model, or models, should be coded in this file

Time unit throughout: years
Compartment unit throughout: patients

"""

import random
from scipy import exp, log
from autumn.base import BaseModel
from curve import make_sigmoidal_curve, make_two_step_curve, scale_up_function
import numpy
import pylab


def label_intersects_tags(label, tags):
    for tag in tags:
        if tag in label:
            return True
    return False


class ConsolidatedModel(BaseModel):

    """
    The transmission dynamic model to underpin all AuTuMN analyses
    Inherits from BaseModel, which is intended to be general to any infectious disease
    All TB-specific methods and structures are contained in this model
    Methods are written to be adaptable to any model structure selected through the __init__ arguments

    The work-flow of the simulation is structured into the following parts:
        1. Defining the model structure
        2. Initialising the compartments
        3. Setting parameters (needs a lot more work)
        4. Calculating derived parameters and setting scale-up functions
        5. Assigning flows from either parameters or functions
        6. Main loop over simulation time-points:
                a. Extracting scale-up variables
                b. Calculating variable flow rates
                c. Calculating diagnostic variables
                    (b. and  c. can be calculated from a compartment values,
                    parameters, scale-up functions or a combination of these)
        7. Calculating the diagnostic solutions
    """

    def __init__(self,
                 n_organ=0,
                 n_strain=0,
                 n_comorbidity=0,
                 is_lowquality=False,
                 is_amplification=False,
                 is_misassignment=False,
                 scenario=None,
                 inputs=None):

        """
        Args:
            n_organ: whether pulmonary status and smear-positive/smear-negative status
                can be included in the model (which apply to all compartments representing active disease)
                0. No subdivision
                1. All patients are smear-positive pulmonary (avoid)
                2. All patients are pulmonary, but smear status can be selected (i.e. smear-pos/smear-neg)
                3. Full stratification into smear-positive, smear-negative and extra-pulmonary
            n_strain: number of types of drug-resistance included (not strains in the strict phylogenetic sense)
                0. No strains included
                1. All TB is DS-TB (avoid)
                2. DS-TB and MDR-TB
                3. DS-TB, MDR-TB and XDR-TB
                (N.B. this may change in future models, which may include isoniazid mono-resistance, etc.)
            n_comorbidity: number of whole-population stratifications, other than age
                0. No population stratification
                1. Entire population is not at increased risk (avoid)
                2. No additional risk factor or HIV
                3. No additional risk factor, HIV or diabetes
            is_lowquality: Boolean of whether to include detections through the private/low-quality sector
            is_amplification: Boolean of whether to include resistance amplification through treatment default
            is_misassignment: Boolean of whether also to incorporate misclassification of patients with drug-resistance
                    to the wrong strain by the high-quality health system
                Avoid amplification=False but misassignment=True (the model should run with both
                amplification and misassignment, but this combination doesn't make sense)
        """

        BaseModel.__init__(self)

        self.loaded_compartments = None

        # Convert inputs to attributes
        self.n_organ = n_organ
        self.n_strain = n_strain
        self.n_comorbidity = n_comorbidity

        # Set time points for integration (model.times now created in base.py)
        self.start_time = inputs['model_constants']['start_time']
        self.end_time = inputs['model_constants']['scenario_end_time']
        self.time_step = inputs['model_constants']['time_step']

        # Set Boolean conditionals for model structure and additional diagnostics
        self.is_lowquality = is_lowquality
        self.is_amplification = is_amplification
        self.is_misassignment = is_misassignment

        self.scenario = scenario

        self.inputs = inputs

        if self.is_misassignment:
            assert self.is_amplification, 'Misassignment requested without amplification'

        # Define model compartmental structure
        # (note that compartment initialisation has now been shifted to base.py)
        self.define_model_structure()

        # Set the few universally fixed and hard-coded parameters
        self.set_universal_parameters()

        # Set other fixed parameters
        self.set_fixed_parameters()

        # Treatment outcomes that will be universal to all models
        # Global TB outcomes of "completion" and "cure" can be considered "_success",
        # "death" is "_death" (of course), "failure" and "default" are considered "_default"
        # and "transfer out" is removed from denominator calculations.
        self.outcomes = ['_success', '_death', '_default']
        self.non_success_outcomes = self.outcomes[1: 3]

    def define_model_structure(self):

        """
        Args:
            All arguments are set through __init__
            Please refer to __init__ method comments above
        """

        # All compartmental disease stages
        self.compartment_types = [
            'susceptible_fully',
            'susceptible_vac',
            'susceptible_treated',
            'latent_early',
            'latent_late',
            'active',
            'detect',
            'missed',
            'treatment_infect',
            'treatment_noninfect']
        if self.is_lowquality: self.compartment_types += ['lowquality']

        # Broader stages for calculating outputs later
        # Now thinking this should eventually go into the plotting module

        # Stages in progression through treatment
        self.treatment_stages = [
            '_infect',
            '_noninfect']

        # Compartments that contribute to force of infection calculations
        self.infectious_tags = [
            'active',
            'missed',
            'detect',
            'treatment_infect']
        if self.is_lowquality: self.infectious_tags += ['lowquality']

        # Select number of organ statuses
        available_organs = [
            '_smearpos',
            '_smearneg',
            '_extrapul']
        if self.n_organ == 0:
            # Need a list of an empty string to be iterable for methods iterating by organ status
            self.organ_status = ['']
        else:
            self.organ_status = available_organs[:self.n_organ]

        # Select number of strains
        self.available_strains = [
            '_ds',
            '_mdr',
            '_xdr']
        if self.n_strain == 0:
            # Need a list of an empty string to be iterable for methods iterating by strain
            self.strains = ['']
        else:
            self.strains = self.available_strains[:self.n_strain]

        # Select number of risk groups
        available_comorbidities = [
            '_nocomorbs',
            '_hiv',
            '_diabetes']
        if self.n_comorbidity == 0:
            # Need a list of an empty string to be iterable for methods iterating by risk group
            self.comorbidities = ['']
        else:
            self.comorbidities = available_comorbidities[:self.n_comorbidity]

        self.define_age_structure()

        self.initial_compartments = {}
        for compartment in self.compartment_types:
            if compartment in self.inputs['model_constants']:
                self.initial_compartments[compartment] \
                    = self.inputs['model_constants'][compartment]

    def initialise_compartments(self):

        # First initialise all compartments to zero
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for compartment in self.compartment_types:

                    # Replicate susceptible for age-groups and comorbidities
                    if 'susceptible' in compartment:
                        self.set_compartment(compartment + comorbidity + agegroup, 0.)

                    # Replicate latent classes for age-groups, comorbidities and strains
                    elif 'latent' in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + comorbidity + agegroup, 0.)

                    # Replicate active classes for age-groups, comorbidities, strains and organ status
                    elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + strain + comorbidity + agegroup, 0.)

                    # Replicate treatment classes for
                    # age-groups, comorbidities, strains, organ status and assigned strain
                    else:
                        for actual_strain_number in range(len(self.strains)):
                            strain = self.strains[actual_strain_number]
                            for organ in self.organ_status:
                                if self.is_misassignment:
                                    for assigned_strain_number in range(len(self.strains)):
                                        self.set_compartment(
                                            compartment + organ + strain +
                                            '_as' + self.strains[assigned_strain_number][1:] +
                                            comorbidity + agegroup, 0.)
                                else:
                                    self.set_compartment(compartment +
                                                         organ + strain + comorbidity + agegroup, 0.)

        # Populate input_compartments
        # Initialise to DS-TB or no strain if single-strain model
        default_start_strain = '_ds'
        if self.strains == ['']: default_start_strain = ''

        # The equal splits will need to be adjusted, but the important thing is not to
        # initialise multiple strains too early, so that MDR-TB doesn't overtake the model

        for compartment in self.compartment_types:
            if compartment in self.initial_compartments:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        if 'susceptible_fully' in compartment:
                            # Split equally by comorbidities and age-groups
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            # Assign all to DS-TB, split equally by comorbidities and age-groups
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment +
                                                     organ + default_start_strain + comorbidity + agegroup,
                                                     self.initial_compartments[compartment]
                                                     / len(self.organ_status)  # Split equally by organ statuses,
                                                     / len(self.comorbidities)  # split equally by comorbidities
                                                     / len(self.agegroups))  # and split equally by age-groups

    def set_universal_parameters(self):

        """
        Sets parameters that should never be changed in any situation,
        i.e. "by definition" parameters (although note that the infectiousness
        of the single infectious compartment for models unstratified by organ
        status is now set in set_fixed_infectious_proportion, because it is
        dependent upon loading some parameters in find_functions_or_params)
        """

        fixed_parameters = {}
        if len(self.organ_status) < 2:
            # Proportion progressing to the only infectious compartment
            # for models unstratified by organ status
            fixed_parameters['epi_prop'] = 1.
        else:
            fixed_parameters = {
                'tb_multiplier_force_smearpos':
                    1.,  # Infectiousness of smear-positive patients
                'tb_multiplier_force_extrapul':
                    0.}  # Infectiousness of extrapulmonary patients

        for parameter in fixed_parameters:
            self.set_parameter(parameter, fixed_parameters[parameter])

    def set_fixed_parameters(self):

        # Set parameters from the data object
        for key, value in self.inputs['model_constants'].items():
            if type(value) == float:
                self.set_parameter(key, value)

    def set_fixed_infectious_proportion(self):

        # Find a multiplier for the proportion of all cases infectious for
        # models unstructured by organ status.
        if len(self.organ_status) < 2:
            self.set_parameter('tb_multiplier_force',
                               self.params['epi_prop_smearpos'] + \
                               self.params['epi_prop_smearneg'] * self.params['tb_multiplier_force_smearneg'])

    def define_age_structure(self):

        # Work out age-groups from list of breakpoints
        self.agegroups = []

        # If age-group breakpoints are supplied
        if len(self.inputs['model_constants']['age_breakpoints']) > 0:
            for i in range(len(self.inputs['model_constants']['age_breakpoints'])):
                if i == 0:

                    # The first age-group
                    self.agegroups += \
                        ['_age0to' + str(self.inputs['model_constants']['age_breakpoints'][i])]
                else:

                    # Middle age-groups
                    self.agegroups += \
                        ['_age' + str(self.inputs['model_constants']['age_breakpoints'][i - 1]) +
                         'to' + str(self.inputs['model_constants']['age_breakpoints'][i])]

            # Last age-group
            self.agegroups += ['_age' + str(self.inputs['model_constants']['age_breakpoints'][len(self.inputs['model_constants']['age_breakpoints']) - 1]) + 'up']

        # Otherwise
        else:
            # List consisting of one empty string required
            # for the methods that iterate over strains
            self.agegroups += ['']

    ############################################################
    # General underlying methods for use by other methods

    def find_outcome_proportions_by_period(
            self, proportion, early_period, total_period):

        """
        Split one outcome proportion (e.g. default, death) over multiple
        periods
        Args:
            proportion: Total proportion to be split
            early_period: Early time period
            total_period: Late time period
        Returns:
            early_proportion: Proportion allocated to early time period
            late_proportion: Proportion allocated to late time period
        """

        if proportion > 1. or proportion < 0.:
            raise Exception('Proportion greater than one or less than zero')
        elif proportion == 1.:
            # This is just to avoid warnings appearing where the proportion
            # is one - this function isn't really intended for this situation,
            # but preferable to avoid both errors and warnings.
            early_proportion = 0.5
        else:
            early_proportion \
                = 1. - exp(log(1. - proportion) * early_period / total_period)
        late_proportion = proportion - early_proportion
        return early_proportion, late_proportion

    def get_constant_or_variable_param(self, param):

        """
        Simple function to look first in vars then params for a parameter value and
        raise an error if the parameter is not found.

        Args:
            param: String for the parameter (should be the same in either vars or params)

        Returns:
            param_value: The value of the parameter

        """

        if param in self.vars:
            param_value = self.vars[param]
        elif param in self.params:
            param_value = self.params[param]
        else:
            raise NameError('Parameter "' + param + '" not found in either vars or params.')

        return param_value

    ##################################################################
    # The master parameter processing and scale-up setting method

    def process_parameters(self):

        """
        Calls all the methods to set parameters and scale-up functions
        The order in which these methods is run is often important
        """

        if len(self.agegroups) > 1: self.find_ageing_rates()

        self.find_treatment_periods()

        self.find_data_for_functions_or_params()

        self.find_amplification_data()

        self.find_functions_or_params()

        self.find_natural_history_params()

        self.set_fixed_infectious_proportion()

    ##################################################################
    # The methods that process_parameters calls to set parameters and
    # scale-up functions

    def find_ageing_rates(self):

        # Calculate ageing rates as the reciprocal of the width of the age bracket
        for agegroup in self.agegroups:
            if 'up' not in agegroup:
                self.set_parameter('ageing_rate' + agegroup, 1. / (
                    float(agegroup[agegroup.find('to') + 2:]) -
                    float(agegroup[agegroup.find('age') + 3: agegroup.find('to')])))

    def find_treatment_periods(self):

        """
        Work out the periods of time spent infectious and non-infectious
        """

        # If the model isn't stratified by strain, use DS-TB time-periods for the single strain
        if self.strains == ['']:
            self.params['tb_timeperiod_infect_ontreatment'] \
                = self.params['tb_timeperiod_infect_ontreatment_ds']
            self.params['tb_timeperiod_treatment'] \
                = self.params['tb_timeperiod_treatment_ds']

        treatment_outcome_types = self.strains
        if len(self.strains) > 1 and self.is_misassignment:
            treatment_outcome_types += ['_inappropriate']

        for strain in treatment_outcome_types:

            # Find the non-infectious periods
            self.set_parameter(
                'tb_timeperiod_noninfect_ontreatment' + strain,
                self.params['tb_timeperiod_treatment' + strain]
                - self.params['tb_timeperiod_infect_ontreatment' + strain])

    def find_irrelevant_time_variants(self):

        # Work out which time-variant parameters are not relevant to this model structure
        irrelevant_time_variants = []
        for time_variant in self.inputs['time_variants'].keys():
            for strain in self.available_strains:
                if strain not in self.strains and strain in time_variant and '_dst' not in time_variant:
                    irrelevant_time_variants += [time_variant]
            # if 'cost' in time_variant:
            #     irrelevant_time_variants += [time_variant]
            if len(self.strains) < 2 and ('line_dst' in time_variant or '_inappropriate' in time_variant):
                irrelevant_time_variants += [time_variant]
            elif len(self.strains) == 2 and 'secondline_dst' in time_variant:
                irrelevant_time_variants += [time_variant]
            elif len(self.strains) == 2 and 'smearneg' in time_variant:
                irrelevant_time_variants += [time_variant]
            if 'lowquality' in time_variant and not self.is_lowquality:
                irrelevant_time_variants += [time_variant]

        return irrelevant_time_variants

    def find_data_for_functions_or_params(self):

        """
        Method to load all the dictionaries to be used in generating scale-up functions to
        a single attribute of the class instance (to avoid creating heaps of functions for
        irrelevant programs)

        Returns:
            Creates self.scaleup_data, a dictionary of the relevant scale-up data for creating
            scale-up functions in set_scaleup_functions.

        """

        # Collect data to generate scale-up functions
        self.scaleup_data = {}

        # Flag the time-variant parameters that aren't relevant
        irrelevant_time_variants = self.find_irrelevant_time_variants()

        # Find the programs that are relevant and load them to the scaleup_data attribute
        for time_variant in self.inputs['time_variants']:
            if time_variant not in irrelevant_time_variants:
                self.scaleup_data[str(time_variant)] = {}
                for i in self.inputs['time_variants'][time_variant]:
                    if i == 'time_variant':
                        self.scaleup_data[str(time_variant)]['time_variant'] = self.inputs['time_variants'][time_variant][i]
                    # For the smoothness parameter
                    elif i == 'smoothness':
                        self.scaleup_data[str(time_variant)]['smoothness'] = self.inputs['time_variants'][time_variant][i]
                    # For years with data percentages
                    elif type(i) == int and 'program_prop_' in time_variant:
                        self.scaleup_data[str(time_variant)][i] = self.inputs['time_variants'][time_variant][i] / 1E2
                    # For years with data not percentages
                    elif type(i) == int:
                        self.scaleup_data[str(time_variant)][i] = self.inputs['time_variants'][time_variant][i]
                    # For scenarios with data percentages
                    elif type(i) == unicode and 'scenario_' + str(self.scenario) in i and 'prop_' in time_variant:
                        self.scaleup_data[str(time_variant)]['scenario'] = \
                            self.inputs['time_variants'][time_variant]['scenario_' + str(self.scenario)] / 1E2
                    # For scenarios with data not percentages
                    elif type(i) == unicode and 'scenario_' + str(self.scenario) in i:
                        self.scaleup_data[str(time_variant)]['scenario'] = \
                            self.inputs['time_variants'][time_variant]['scenario_' + str(self.scenario)]

    def find_amplification_data(self):

        # Add dictionary for the amplification proportion scale-up (if relevant)
        if len(self.strains) > 1:
            self.scaleup_data['epi_prop_amplification'] = \
                {self.params['start_mdr_introduce_period']:
                     0.,
                 self.params['end_mdr_introduce_period']:
                     self.params['tb_prop_amplification'],
                 'time_variant':
                    'yes'}

    def find_functions_or_params(self):

        # Define scale-up functions from these datasets
        for param in self.scaleup_data:

            time_variant = self.scaleup_data[param].pop('time_variant')

            if param == 'epi_prop_smearpos':
                if time_variant == 'yes':
                    self.is_organvariation = True
                else:
                    self.is_organvariation = False

            if time_variant == 'yes':

                # Extract and remove the smoothness parameter from the dictionary
                if 'smoothness' in self.scaleup_data[param]:
                    smoothness = self.scaleup_data[param].pop('smoothness')
                else:
                    smoothness = self.inputs['model_constants']['default_smoothness']

                # If the parameter is being modified for the scenario being run
                if 'scenario' in self.scaleup_data[param]:
                    scenario = [self.params['scenario_full_time'],
                                self.scaleup_data[param].pop('scenario')]
                else:
                    scenario = None

                # Upper bound depends on whether the parameter is a proportion
                if 'prop' in param:
                    upper_bound = 1.
                else:
                    upper_bound = 1E7

                # Calculate the scaling function
                self.set_scaleup_fn(param,
                                    scale_up_function(self.scaleup_data[param].keys(),
                                                      self.scaleup_data[param].values(),
                                                      self.inputs['model_constants']['fitting_method'],
                                                      smoothness,
                                                      bound_low=0.,
                                                      bound_up=upper_bound,
                                                      intervention_end=scenario,
                                                      intervention_start_date=self.params['scenario_start_time']))

            # If no is selected in the time variant column
            elif time_variant == 'no':

                # Get rid of smoothness, which isn't relevant
                if 'smoothness' in self.scaleup_data[param]:
                    del self.scaleup_data[param]['smoothness']

                # Set as a constant parameter
                self.set_parameter(param,
                                   self.scaleup_data[param][max(self.scaleup_data[param])])

                # Note that the 'demo_life_expectancy' parameter has to be given this name
                # and base.py will then calculate population death rates automatically.

        if not self.is_organvariation:
            self.set_parameter('epi_prop_extrapul',
                               1. - self.params['epi_prop_smearpos'] - self.params['epi_prop_smearneg'])

    def find_natural_history_params(self):

        # If extrapulmonary case-fatality not stated, use smear-negative case-fatality
        if 'tb_prop_casefatality_untreated_extrapul' not in self.params:
            self.set_parameter(
                'tb_prop_casefatality_untreated_extrapul',
                self.params['tb_prop_casefatality_untreated_smearneg'])

        # Overall early progression rate
        self.set_parameter('tb_rate_early_progression',
                           self.params['tb_prop_early_progression']
                           / self.params['tb_timeperiod_early_latent'])

        # Stabilisation rate
        self.set_parameter('tb_rate_stabilise',
                           (1. - self.params['tb_prop_early_progression'])
                           / self.params['tb_timeperiod_early_latent'])

        # Adjust overall death and recovery rates by organ status
        for organ in self.organ_status:
            self.set_parameter(
                'tb_rate_death' + organ,
                self.params['tb_prop_casefatality_untreated' + organ]
                / self.params['tb_timeperiod_activeuntreated'])
            self.set_parameter(
                'tb_rate_recover' + organ,
                (1 - self.params['tb_prop_casefatality_untreated' + organ])
                / self.params['tb_timeperiod_activeuntreated'])

        if not self.is_organvariation:
            for organ in self.organ_status:
                for timing in ['_early', '_late']:
                    self.set_parameter('tb_rate' + timing + '_progression' + organ,
                                       self.params['tb_rate' + timing + '_progression']
                                       * self.params['epi_prop' + organ])

    ##################################################################
    # Methods that calculate variables to be used in calculating flows
    # Note: all scaleup_fns are calculated and put into self.vars before
    # calculate_vars
    # I think we have to put any calculations that are dependent upon vars
    # into this section

    def calculate_vars(self):

        """
        The master method that calls all the other methods for the calculations of
        variable rates
        """

        self.vars['population'] = sum(self.compartments.values())

        self.calculate_birth_rates_vars()

        self.calculate_force_infection_vars()

        self.calculate_progression_vars()

        self.calculate_detect_missed_vars()

        self.calculate_proportionate_detection_vars()

        if self.is_lowquality: self.calculate_lowquality_detection_vars()

        self.calculate_await_treatment_var()

        self.calculate_treatment_rates_vars()

        self.calculate_ipt_rate()

    def calculate_birth_rates_vars(self):

        """
        Calculate birth rates into vaccinated and unvaccinated compartments
        """

        # Get the parameters depending on whether constant or time variant
        rate_birth = self.get_constant_or_variable_param('demo_rate_birth') / 1E3
        prop_vacc = self.get_constant_or_variable_param('program_prop_vaccination')

        # Calculate the birth rates by compartment
        self.vars['births_unvac'] = \
            (1. - prop_vacc) \
            * rate_birth \
            * self.vars['population'] \
            / len(self.comorbidities)
        self.vars['births_vac'] = \
            prop_vacc \
            * rate_birth \
            * self.vars['population'] \
            / len(self.comorbidities)

    def calculate_force_infection_vars(self):

        # Calculate force of infection by strain,
        # incorporating partial immunity and infectiousness
        for strain in self.strains:

            # Initialise infectious population to zero
            self.vars['infectious_population' + strain] = 0.
            for organ in self.organ_status:
                for label in self.labels:

                    # If we haven't reached the part of the model divided by organ status
                    # and we're not doing the organ status of interest.
                    if organ not in label and organ != '':
                        continue

                    # If we haven't reached the part of the model divided by strain
                    # and we're not doing the strain of interest.
                    if strain not in label and strain != '':
                        continue

                    # If the compartment is non-infectious
                    if label_intersects_tags(label, self.infectious_tags):

                        self.vars['infectious_population' + strain] += \
                            self.params['tb_multiplier_force' + organ] \
                            * self.compartments[label]

            # Calculate non-immune force of infection
            self.vars['rate_force' + strain] = \
                self.params['tb_n_contact'] \
                * self.vars['infectious_population' + strain] \
                / self.vars['population']

            # Adjust for immunity
            self.vars['rate_force_vacc' + strain] = \
                self.params['tb_multiplier_bcg_protection'] \
                * self.vars['rate_force' + strain]
            self.vars['rate_force_latent' + strain] = \
                self.params['tb_multiplier_latency_protection'] \
                * self.vars['rate_force' + strain]

    def calculate_progression_vars(self):

        """
        Calculate vars for the remainder of progressions.
        Note that the vars for the smear-positive and smear-negative proportions
        have already been calculated. However, all progressions have to go somewhere,
        so need to calculate the remaining proportions.
        """

        if self.is_organvariation:

            # If unstratified (self.organ_status should have length 0, but length 1 OK)
            if len(self.organ_status) < 2:
                self.vars['epi_prop'] = 1.

            # Stratified into smear-positive and smear-negative
            elif len(self.organ_status) == 2:
                self.vars['epi_prop_smearneg'] = \
                    1. - self.vars['epi_prop_smearpos']

            # Fully stratified into smear-positive, smear-negative and extra-pulmonary
            elif len(self.organ_status) > 2:
                self.vars['epi_prop_extrapul'] = \
                    1. - self.vars['epi_prop_smearpos'] - self.vars['epi_prop_smearneg']

            # Determine variable progression rates
            for organ in self.organ_status:
                for timing in ['_early', '_late']:
                    self.vars['tb_rate' + timing + '_progression' + organ] \
                        = self.vars['epi_prop' + organ] * self.params['tb_rate' + timing + '_progression']

    def calculate_detect_missed_vars(self):

        """"
        Calculate rates of detection and failure of detection
        from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here)

        Derived from original formulas of by solving the simultaneous equations:
          algorithm sensitivity = detection rate / (detection rate + missed rate)
          - and -
          detection proportion = detection rate / (detection rate + spont recover rate + tb death rate + natural death rate)
        """

        # Detection
        # Note that all organ types are assumed to have the same untreated active
        # sojourn time, so any organ status can be arbitrarily selected (here the first, or smear-positive)

        detect_prop = self.get_constant_or_variable_param('program_prop_detect')

        # If no division by zero
        if self.vars['program_prop_algorithm_sensitivity'] > 0.:

            # Detections
            self.vars['program_rate_detect'] = \
                - detect_prop \
                * (self.params['tb_rate_recover' + self.organ_status[0]] +
                   self.params['tb_rate_death' + self.organ_status[0]] +
                   1. / self.params['demo_life_expectancy']) \
                / (detect_prop - 1.)

            # Missed
            self.vars['program_rate_missed'] = \
                self.vars['program_rate_detect'] \
                * (1. - self.vars['program_prop_algorithm_sensitivity']) \
                / self.vars['program_prop_algorithm_sensitivity']

        # Otherwise just assign detection and missed rates to zero
        else:
            self.vars['program_rate_detect'] = 0.
            self.vars['program_rate_missed'] = 0.

        # Repeat for each strain
        for strain in self.strains:
            for programmatic_rate in ['_detect', '_missed']:
                self.vars['program_rate' + programmatic_rate + strain] = \
                    self.vars['program_rate' + programmatic_rate]

    def calculate_await_treatment_var(self):

        """
        Take the reciprocal of the waiting times to calculate the flow rate to start
        treatment after detection.
        Note that the default behaviour for a single strain model is to use the
        waiting time for smear-positive patients.
        Also weight the time period
        """

        if len(self.organ_status) < 2:
            self.vars['program_rate_start_treatment'] = \
                1. / self.vars['program_timeperiod_await_treatment_smearpos']
        else:
            for organ in self.organ_status:
                if organ == '_smearneg':
                    self.vars['program_rate_start_treatment_smearneg'] = \
                        1. / (self.vars['program_timeperiod_await_treatment_smearneg'] * (1. - self.vars['program_prop_xpert'])
                            + self.params['program_timeperiod_await_treatment_smearneg_xpert'] * self.vars['program_prop_xpert'])
                else:
                    self.vars['program_rate_start_treatment' + organ] = \
                        1. / self.vars['program_timeperiod_await_treatment' + organ]

    def calculate_lowquality_detection_vars(self):

        """
        Calculate rate of entry to low-quality care,
        form the proportion of treatment administered in low-quality sector

        Note that this now means that the case detection proportion only
        applies to those with access to care and so that proportion of all
        cases isn't actually detected
        """

        self.vars['program_rate_enterlowquality'] = \
            self.vars['program_rate_detect'] \
            * self.vars['program_prop_lowquality'] \
            / (1. - self.vars['program_prop_lowquality'])

    def calculate_proportionate_detection_vars(self):

        """
        Calculate the proportions of patients assigned to each strain
        """

        # With misassignment:

        # Note that second-line DST availability refers to the proportion
        # of those with first-line DST who also have second-line DST available
        # (rather than the proportion of all presentations with second line
        # treatment available)
        if self.is_misassignment:

            # DS-TB
            self.vars['program_rate_detect_ds_asds'] = \
                self.vars['program_rate_detect']
            self.vars['program_rate_detect_ds_asmdr'] = 0.
            self.vars['program_rate_detect_ds_asxdr'] = 0.

            # MDR-TB
            self.vars['program_rate_detect_mdr_asds'] = \
                (1. - self.vars['program_prop_firstline_dst']) \
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_mdr_asmdr'] = \
                self.vars['program_prop_firstline_dst'] \
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_mdr_asxdr'] = 0.

            # XDR-TB
            self.vars['program_rate_detect_xdr_asds'] = \
                (1. - self.vars['program_prop_firstline_dst']) \
                 * self.vars['program_rate_detect']
            self.vars['program_rate_detect_xdr_asmdr'] = \
                self.vars['program_prop_firstline_dst'] \
                * (1. - self.vars['program_prop_secondline_dst'])\
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_xdr_asxdr'] = \
                self.vars['program_prop_firstline_dst'] \
                * self.vars['program_prop_secondline_dst'] \
                * self.vars['program_rate_detect']

        # Without misassignment:
        else:
            for strain in self.strains:
                self.vars['program_rate_detect' + strain + '_as'+strain[1:]] = \
                    self.vars['program_rate_detect']

    def calculate_treatment_rates_vars(self):

        # May need to adjust this - a bit of a patch for now
        treatments = self.strains
        if len(self.strains) > 1:
            treatments += ['_inappropriate']

        for strain in treatments:

            self.vars['program_prop_treatment_default' + strain] \
                = 1. \
                  - self.vars['program_prop_treatment_success' + strain] \
                  - self.vars['program_prop_treatment_death' + strain]

            # Find the proportion of deaths/defaults during the infectious and non-infectious stages
            for outcome in self.non_success_outcomes:
                early_proportion, late_proportion = self.find_outcome_proportions_by_period(
                    self.vars['program_prop_treatment' + outcome + strain],
                    self.params['tb_timeperiod_infect_ontreatment' + strain],
                    self.params['tb_timeperiod_treatment' + strain])
                self.vars['program_prop_treatment' + outcome + '_infect' + strain] = early_proportion
                self.vars['program_prop_treatment' + outcome + '_noninfect' + strain] = late_proportion

            # Find the success proportions
            for treatment_stage in self.treatment_stages:
                self.vars['program_prop_treatment_success' + treatment_stage + strain] = \
                    1. - self.vars['program_prop_treatment_default' + treatment_stage + strain] \
                    - self.vars['program_prop_treatment_death' + treatment_stage + strain]

                # Find the corresponding rates from the proportions
                for outcome in self.outcomes:
                    self.vars['program_rate' + outcome + treatment_stage + strain] = \
                        1. / self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                        * self.vars['program_prop_treatment' + outcome + treatment_stage + strain]
                if self.is_amplification:
                    self.vars['program_rate_default' + treatment_stage + '_amplify' + strain] = \
                        self.vars['program_rate_default' + treatment_stage + strain] \
                        * self.vars['epi_prop_amplification']
                    self.vars['program_rate_default' + treatment_stage + '_noamplify' + strain] = \
                        self.vars['program_rate_default' + treatment_stage + strain] \
                        * (1. - self.vars['epi_prop_amplification'])

    def calculate_ipt_rate(self):

        """
        Calculate the number of persons starting treatment for IPT, which is linked to the number
        of patients starting treatment.
        This code structure echoes the code in set_variable_programmatic_flows
        """

        prop_ipt = self.get_constant_or_variable_param('program_prop_ipt')

        # This parameter is the number of persons effectively treated with IPT for each
        # patient started on treatment for active disease.
        prevented_cases_per_patient_starting_treatment = prop_ipt \
                                                         * (self.inputs['model_constants']['demo_household_size'] - 1.) \
                                                         * self.inputs['model_constants']['tb_prop_contacts_infected'] \
                                                         * self.inputs['model_constants']['tb_prop_ltbi_test_sensitivity'] \
                                                         * self.inputs['model_constants']['tb_prop_ipt_effectiveness']
        self.vars['ipt_commencements'] = 0.
        for strain in self.strains:
            for agegroup in self.agegroups:
                for comorbidity in self.comorbidities:
                    for organ in self.organ_status:
                        # Currently only applying to patients with drug-susceptible TB,
                        # which is presumed to be all patients if the model is unstratified by strain
                        # and only '_ds' if the model is stratified.
                        if 'dr' not in strain and organ == '_smearpos':
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    if 'dr' not in assigned_strain:
                                        self.vars['ipt_commencements'] += \
                                            self.compartments['detect' + organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup] * \
                                            self.vars['program_rate_start_treatment' + organ] * \
                                            prevented_cases_per_patient_starting_treatment
                            else:
                                self.vars['ipt_commencements'] += \
                                    self.compartments['detect' + organ + strain + comorbidity + agegroup] * \
                                    self.vars['program_rate_start_treatment' + organ] * \
                                    prevented_cases_per_patient_starting_treatment

    ##################################################################
    # Methods that calculate the flows of all the compartments

    def set_flows(self):

        """
        Call all the rate setting methods
        """

        self.set_birth_flows()

        if len(self.agegroups) > 0: self.set_ageing_flows()

        self.set_infection_flows()

        self.set_progression_flows()

        self.set_natural_history_flows()

        self.set_fixed_programmatic_flows()

        self.set_variable_programmatic_flows()

        self.set_detection_flows()

        self.set_treatment_flows()

        self.set_ipt_flows()

    def set_birth_flows(self):

        # Set birth flows (currently evenly distributed between comorbidities)
        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                'susceptible_fully' + comorbidity + self.agegroups[0], 'births_unvac')
            self.set_var_entry_rate_flow(
                'susceptible_vac' + comorbidity + self.agegroups[0], 'births_vac')

    def set_ageing_flows(self):

        # Set simple ageing flows for any number of strata
        for label in self.labels:
            for number_agegroup in range(len(self.agegroups)):
                if self.agegroups[number_agegroup] in label and number_agegroup < len(self.agegroups) - 1:
                    self.set_fixed_transfer_rate_flow(
                        label, label[0: label.find('_age')] + self.agegroups[number_agegroup + 1],
                               'ageing_rate' + self.agegroups[number_agegroup])

    def set_infection_flows(self):

        # Set force of infection flows
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                    # Fully susceptible
                    self.set_var_transfer_rate_flow(
                        'susceptible_fully' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force' + strain)

                    # Partially immune
                    self.set_var_transfer_rate_flow(
                        'susceptible_vac' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vacc' + strain)
                    self.set_var_transfer_rate_flow(
                        'susceptible_treated' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vacc' + strain)
                    self.set_var_transfer_rate_flow(
                        'latent_late' + strain + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_latent' + strain)

    def set_progression_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                        # Stabilisation
                        self.set_fixed_transfer_rate_flow(
                            'latent_early' + strain + comorbidity + agegroup,
                            'latent_late' + strain + comorbidity + agegroup,
                            'tb_rate_stabilise')

                        for organ in self.organ_status:

                            # If organ scale-ups available, set flows as variable
                            # (if epi_prop_smearpos is in self.scaleup_fns, then epi_prop_smearneg
                            # should be too)
                            if self.is_organvariation:
                                # Early progression
                                self.set_var_transfer_rate_flow(
                                    'latent_early' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_early_progression' + organ)

                                # Late progression
                                self.set_var_transfer_rate_flow(
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_late_progression' + organ)

                            # Otherwise, set fixed flows
                            else:
                                # Early progression
                                self.set_fixed_transfer_rate_flow(
                                    'latent_early' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_early_progression' + organ)

                                # Late progression
                                self.set_fixed_transfer_rate_flow(
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_late_progression' + organ)

    def set_natural_history_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Recovery, base compartments
                            self.set_fixed_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'latent_late' + strain + comorbidity + agegroup,
                                'tb_rate_recover' + organ)
                            self.set_fixed_transfer_rate_flow(
                                'missed' + organ + strain + comorbidity + agegroup,
                                'latent_late' + strain + comorbidity + agegroup,
                                'tb_rate_recover' + organ)

                            # Death, base compartments
                            self.set_fixed_infection_death_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'tb_rate_death' + organ)
                            self.set_fixed_infection_death_rate_flow(
                                'missed' + organ + strain + comorbidity + agegroup,
                                'tb_rate_death' + organ)

                            # Extra low-quality compartments
                            if self.is_lowquality:
                                self.set_fixed_transfer_rate_flow(
                                    'lowquality' + organ + strain + comorbidity + agegroup,
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'tb_rate_recover' + organ)
                                self.set_fixed_infection_death_rate_flow(
                                    'lowquality' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_death' + organ)

                            # Detected, with misassignment
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.set_fixed_infection_death_rate_flow(
                                        'detect' +
                                        organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                        'tb_rate_death' + organ)
                                    self.set_fixed_transfer_rate_flow(
                                        'detect' +
                                        organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                        'latent_late' + strain + comorbidity + agegroup,
                                        'tb_rate_recover' + organ)

                            # Detected, without misassignment
                            else:
                                self.set_fixed_transfer_rate_flow(
                                    'detect' + organ + strain + comorbidity + agegroup,
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'tb_rate_recover' + organ)
                                self.set_fixed_infection_death_rate_flow(
                                    'detect' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_death' + organ)

    def set_fixed_programmatic_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Re-start presenting after a missed diagnosis
                        self.set_fixed_transfer_rate_flow(
                            'missed' + organ + strain + comorbidity + agegroup,
                            'active' + organ + strain + comorbidity + agegroup,
                            'program_rate_restart_presenting')

                        # Give up on the hopeless low-quality health system
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'active' + organ + strain + comorbidity + agegroup,
                                'program_rate_leavelowquality')

    def set_detection_flows(self):

        """
        Set previously calculated detection rates
        Either assuming everyone is correctly identified if misassignment not permitted
        or with proportional misassignment
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for organ in self.organ_status:
                    for actual_strain_number in range(len(self.strains)):
                        strain = self.strains[actual_strain_number]

                        # With misassignment
                        if self.is_misassignment:
                            for assigned_strain_number in range(len(self.strains)):
                                as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]
                                # If the strain is equally or more resistant than its assignment
                                if actual_strain_number >= assigned_strain_number:
                                    self.set_var_transfer_rate_flow(
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'detect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'program_rate_detect' + strain + as_assigned_strain)

                        # Without misassignment - everyone is correctly identified by strain
                        else:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'detect' + organ + strain + comorbidity + agegroup,
                                'program_rate_detect')

    def set_variable_programmatic_flows(self):

        # Set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        self.set_var_transfer_rate_flow(
                            'active' + organ + strain + comorbidity + agegroup,
                            'missed' + organ + strain + comorbidity + agegroup,
                            'program_rate_missed')
                        # Detection, with and without misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                # Following line only for models incorporating mis-assignment
                                self.set_var_transfer_rate_flow(
                                    'detect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'treatment_infect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'program_rate_start_treatment' + organ)
                        else:
                            # Following line is the currently running line (without mis-assignment)
                            self.set_var_transfer_rate_flow(
                                'detect' + organ + strain + comorbidity + agegroup,
                                'treatment_infect' + organ + strain + comorbidity + agegroup,
                                'program_rate_start_treatment' + organ)

                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'program_rate_enterlowquality')

    def set_treatment_flows(self):

        """
        Set rates of progression through treatment stages
        Accommodates with and without amplification, and with and without misassignment
        """
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for organ in self.organ_status:

                    for actual_strain_number in range(len(self.strains)):
                        strain = self.strains[actual_strain_number]
                        if self.is_misassignment:
                            assignment_strains = range(len(self.strains))
                        else:
                            assignment_strains = ['']
                        for assigned_strain_number in assignment_strains:
                            if self.is_misassignment:
                                as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]

                                # Which treatment parameters to use - for the strain or for inappropriate treatment
                                if actual_strain_number > assigned_strain_number:
                                    strain_or_inappropriate = '_inappropriate'
                                else:
                                    strain_or_inappropriate = self.strains[assigned_strain_number]

                            else:
                                as_assigned_strain = ''
                                strain_or_inappropriate = self.strains[actual_strain_number]

                            # Success at either treatment stage
                            self.set_var_transfer_rate_flow(
                                'treatment_infect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'program_rate_success_infect' + strain_or_inappropriate)
                            self.set_var_transfer_rate_flow(
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'susceptible_treated' + comorbidity + agegroup,
                                'program_rate_success_noninfect' + strain_or_inappropriate)

                            # Rates of death on treatment
                            for treatment_stage in self.treatment_stages:
                                self.set_var_infection_death_rate_flow(
                                    'treatment' +
                                    treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                    'program_rate_death' + treatment_stage + strain_or_inappropriate)

                            # Default
                            for treatment_stage in self.treatment_stages:

                                # If it's either the most resistant strain available or amplification is not active:
                                if actual_strain_number == len(self.strains) - 1 or not self.is_amplification:
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + strain_or_inappropriate)

                                # Otherwise
                                else:
                                    amplify_to_strain = self.strains[actual_strain_number + 1]
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_noamplify' + strain_or_inappropriate)
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + amplify_to_strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_amplify' + strain_or_inappropriate)

    def set_ipt_flows(self):

        """
        Sets a flow from the early latent compartment to the partially immune susceptible compartment
        that is determined by report_numbers_starting_treatment above and is not linked to the
        'from_label' compartment.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    self.set_linked_transfer_rate_flow('latent_early' + strain + comorbidity + agegroup,
                                                       'susceptible_vac' + strain + comorbidity + agegroup,
                                                       'ipt_commencements')

    ##################################################################
    # Methods that calculate the output vars and diagnostic properties

    def calculate_output_vars(self):

        """
        Method similarly structured to calculate_output_vars,
        just replicated by strains
        """

        # Initialise dictionaries
        rate_incidence = {}
        rate_mortality = {}
        rate_notifications = {}

        for strain in self.strains:

            # Initialise scalars
            rate_incidence[strain] = 0.
            rate_mortality[strain] = 0.
            rate_notifications[strain] = 0.

            # Incidence
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            for from_label, to_label, rate in self.fixed_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * rate

            self.vars['incidence' + strain] \
                = rate_incidence[strain] / self.vars['population'] * 1E5

            # Notifications
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'active' in from_label and 'detect' in to_label and strain in from_label:
                    rate_notifications[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars['notifications' + strain] \
                = rate_notifications[strain]

            # Mortality
            for from_label, rate in self.fixed_infection_death_rate_flows:
                # Under-reporting factor included for those deaths not occurring on treatment
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * rate \
                            * self.params['program_prop_death_reporting']
            for from_label, rate in self.var_infection_death_rate_flows:
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars['mortality' + strain] \
                = rate_mortality[strain] / self.vars['population'] * 1E5

        # Prevalence
        for strain in self.strains:
            self.vars['prevalence' + strain] = 0.
            for label in self.labels:
                if 'susceptible' not in label and \
                                'latent' not in label and strain in label:
                    self.vars['prevalence' + strain] \
                        += (self.compartments[label]
                        / self.vars['population'] * 1E5)

        # Summing MDR and XDR to get the total of all MDRs
        if len(self.strains) > 1:
            rate_incidence['all_mdr_strains'] = 0.
            if len(self.strains) > 1:
                for actual_strain_number in range(len(self.strains)):
                    strain = self.strains[actual_strain_number]
                    if actual_strain_number > 0:
                        rate_incidence['all_mdr_strains'] \
                            += rate_incidence[strain]
            self.vars['all_mdr_strains'] \
                = rate_incidence['all_mdr_strains'] / self.vars['population'] * 1E5
            # Convert to percentage
            self.vars['proportion_mdr'] \
                = self.vars['all_mdr_strains'] / self.vars['incidence'] * 1E2

    ##################################################################
    # Methods to call base integration function depending on the type of
    # integration required

    def integrate(self):
        min_dt = 0.05
        if self.inputs['model_constants']['integration'] == 'explicit':
            self.integrate_explicit(min_dt)
        elif self.inputs['model_constants']['integration'] == 'scipy':
            self.integrate_scipy(min_dt)
        elif self.inputs['model_constants']['integration'] == 'runge_kutta':
            self.integrate_runge_kutta(min_dt)


