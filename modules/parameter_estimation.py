# -*- coding: utf-8 -*-
"""
The following module provides the framework for setting parameter objects -
both parameters to be used in the model itself, as well as references to
evidence that can subsequently be used to determine these parameter
distributions.
Need to check variance calculations - e.g. whether the gamma variance
calculation is based on the variance or the standard deviation.
Created on Thu Nov 12 13:45:23 2015

@author: JTrauer
"""

import os
import sys
import numpy
from scipy.stats import beta, gamma, norm, truncnorm
import matplotlib.pyplot as pyplot


class AllEvidence(type):
    def __iter__(evidencepiece):
        return iter(evidencepiece.evidence_register)

class Evidence:
    """ Object to summarise evidence for use in parameter estimation """
    __metaclass__ = AllEvidence
    evidence_register = []
    def __init__(self, source, parameter, point_estimate, confidence_interval,
                 evidence_fullname, explanation_evidence,
                 reference):
        self.evidence_register.append(self)
        self.estimate = point_estimate
        self.interval = confidence_interval
        self.name = source
        self.fullname = evidence_fullname
        self.reference = reference
        self.explanation = explanation_evidence
        if len(confidence_interval) == 2:
            self.interval_text = ('(' + str(self.interval[0]) + ' - ' +
               str(self.interval[1]) + ')')
        elif len(confidence_interval) < 2:
            self.interval_text = 'No confidence interval available from study'
        self.text = {'Title': self.fullname,
            'Reference': self.reference,
            'Point estimate': str(self.estimate),
            'Confidence interval': self.interval_text,
            'Explanation': self.explanation}
        self.attributes_ordered = ['Title', 'Point estimate',
                                   'Confidence interval', 'Explanation']
    def open_pdf(self):
        current_dir = os.path.dirname(__file__)
        location = os.path.join(current_dir, '..', 'evidence',
                                self.name + '.pdf')
        os.startfile(location)

#______________________________________________________________________________

class AllParameters(type):
    def __iter__(parameterinstance):
        return(iter(parameterinstance.parameter_register))

class Parameter:
    """"  Initialises parameters with distributions prior to model runs """
    __metaclass__ = AllParameters
    parameter_register = []
    def __init__(self, name, parameter_name, parameter_type,
                 distribution, prior_estimate, spread, limits,
                 model_implementation):
        self.parameter_register.append(self)
        self.name = name
        self.parameter_name = parameter_name
        self.parameter_type = parameter_type
        self.model_implementation = model_implementation
        available_types = ['proportion',
                           'rate',
                           'timeperiod',
                           'multiplier']
        assert self.parameter_type in available_types
        if self.parameter_type == 'proportion':
            assert len(self.model_implementation) == 2
            self.implementation_description = ('Numerator is ' +
            str(self.model_implementation[0]) + ' and denominator is ' +
            str(self.model_implementation[1]) + '\n\n')
        elif self.parameter_type == 'rate':
            assert len(self.model_implementation) == 2
            self.implementation_description = ('From compartment ' +
            self.model_implementation[0] + ' to compartment ' +
            self.model_implementation[1] + '\n\n')
        elif self.parameter_type == 'timeperiod':
            assert len(self.model_implementation) == 1
            self.implementation_description = ('Time spent in ' +
            self.model_implementation[0])
        elif self.parameter_type == 'multiplier':
            assert len(self.model_implementation) == 1
            self.implementation_description = ('Parameter to be multiplied is ' +
            self.model_implementation[0])
        self.distribution = distribution
        available_distributions = ['beta_symmetric_params2',
                                   'beta_full_range', 'gamma',
                                   'normal_unlimited', 'normal_positive',
                                   'normal_truncated']
        assert distribution in available_distributions, \
            'Distribution not available'
        self.prior_estimate = prior_estimate
        if len(spread) == 0:
            if self.parameter_type == 'proportion':
                if self.prior_estimate < 0.5:
                    self.spread = prior_estimate / 2.
                else:
                    self.spread = (1 - prior_estimate) / 2.
            else:
                 self.spread = prior_estimate / 2.
        elif len(spread) == 1:
            self.spread = spread[0]
        elif len(spread) == 2:
            self.spread = (spread[1] - spread[0]) / 4.
        self.limits = limits
        assert len(self.limits) <= 2, 'Too many limits provided'
        if len(self.limits) == 0:
            self.limit_text = 'No additional limits applied'
        elif len(self.limits) == 1:
            self.limit_text = ('One additional limit set at ' +
            str(self.limits[0]))
        elif len(self.limits) == 2:
            self.limit_text = ('Two additional limits set at ' +
            str(self.limits[0]) + ' and ' + str(self.limits[1]))
        self.text = {'Title': self.parameter_name,
                     'Type': self.parameter_type,
                     'Estimate': str(self.prior_estimate),
                     'Spread': str(self.spread),
                     'Limits': str(self.limit_text),
                     'Implementation': str(self.implementation_description),
                     'Distribution': str(self.distribution)}
        self.attributes_ordered = ['Title', 'Type', 'Distribution',
                                   'Estimate', 'Spread', 'Limits',
                                   'Implementation']

    def calculate_prior(self):
        if self.distribution == 'beta_symmetric_params2':
            self.xvalues = numpy.arange(0., 1., 1e-3)
            self.initiate_distributions()
            self.beta_symmetric_params2()
            self.x_max_forgraph = self.prior_estimate * 2.
        elif self.distribution == 'beta_full_range':
            self.xvalues = numpy.arange(0., 1., 1e-3)
            self.initiate_distributions()
            self.beta_full_range()
            self.x_max_forgraph = 1.
        elif self.distribution == 'gamma':
            self.xvalues = numpy.arange(0., self.prior_estimate * 3.,
                                        self.prior_estimate / 1e2)
            self.initiate_distributions()
            self.gamma()
            self.x_max_forgraph = self.prior_estimate * 3.
        elif self.distribution == 'normal_unlimited':
            self.xvalues = numpy.arange(0., self.prior_estimate * 5, 1e-3)
            self.initiate_distributions()
            self.normal_unlimited()
            self.x_max_forgraph = self.prior_estimate * 2.
        elif self.distribution == 'normal_positive':
            self.xvalues = numpy.arange(0., self.prior_estimate * 5, 1e-3)
            self.initiate_distributions()
            self.normal_positive()
            self.x_max_forgraph = self.prior_estimate * 2.
        elif self.distribution == 'normal_truncated':
            assert isinstance(self.limits, list), 'List of two limits required'
            assert len(self.limits) == 2, 'List of two limits required'
            self.xvalues = numpy.arange(0., self.prior_estimate * 5, 1e-3)
            self.initiate_distributions()
            self.normal_truncated()
            self.x_max_forgraph = self.prior_estimate * 2.

    def initiate_distributions(self):
        self.prior_pdf = [0] * len(self.xvalues)
        self.prior_cdf = [0] * len(self.xvalues)

    def beta_symmetric_params2(self):
        assert self.prior_estimate > self.spread, \
            'Spread greater than prior estimate, negative values will result'
        self.distribution_description = ('Symmetric beta distribution with ' +
            'alpha parameter = beta parameter = 2')
        if self.spread == 0:
            self.spread = self.prior_estimate / 2.
            print('No spread given, so prior_estimate used for range')
        self.lower_limit = self.prior_estimate - self.spread
        self.upper_limit = self.prior_estimate + self.spread
        self.beta_param_alpha = 2
        self.beta_param_beta = 2
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] \
                = self.beta_symmetric_params2_pdf(self.xvalues[i])
            self.prior_cdf[i] \
                = self.beta_symmetric_params2_cdf(self.xvalues[i])

    def beta_symmetric_params2_pdf(self, xvalue):
        transformed_value = (xvalue - self.lower_limit) \
                        / (self.upper_limit - self.lower_limit)
        if transformed_value > 0 and transformed_value < 1:
            beta_symmetric_params2_pdf = beta.pdf(transformed_value,
                                                  self.beta_param_alpha,
                                                  self.beta_param_beta)
        else:
            beta_symmetric_params2_pdf = 0
        return beta_symmetric_params2_pdf

    def beta_symmetric_params2_cdf(self, xvalue):
        transformed_value = (xvalue - self.lower_limit) \
                        / (self.upper_limit - self.lower_limit)
        if transformed_value > 0 and transformed_value < 1:
            beta_symmetric_params2_cdf = beta.cdf(transformed_value,
                                                  self.beta_param_alpha,
                                                  self.beta_param_beta)
        elif transformed_value >= 1:
            beta_symmetric_params2_cdf = 1
        else:
            beta_symmetric_params2_cdf = 0
        return beta_symmetric_params2_cdf

    def beta_full_range(self):
        self.distribution_description = ('Beta distribution with parameters ' +
            'determined from expectation and spread values')
        self.lower_limit = 0
        self.upper_limit = 1
        self.beta_param_alpha \
            = - self.prior_estimate \
                * (self.spread ** 2 + self.prior_estimate ** 2 - self.prior_estimate) \
                / (self.spread ** 2)
        self.beta_param_beta \
           = (self.spread ** 2 + self.prior_estimate ** 2 - self.prior_estimate) \
               * (self.prior_estimate - 1) / (self.spread ** 2)
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] = self.beta_full_range_pdf(self.xvalues[i])
            self.prior_cdf[i] = self.beta_full_range_cdf(self.xvalues[i])

    def beta_full_range_pdf(self, xvalue):
        beta_full_range_pdf = beta.pdf(xvalue, self.beta_param_alpha,
                                       self.beta_param_beta)
        return beta_full_range_pdf

    def beta_full_range_cdf(self, xvalue):
        beta_full_range_cdf = beta.cdf(xvalue, self.beta_param_alpha,
                                       self.beta_param_beta)
        return beta_full_range_cdf

    def gamma(self):
        self.distribution_description = ('Gamma distribution with parameters ' +
            'determined from expectation and spread values')
        self.lower_limit = 0
        self.upper_limit = 'No upper limit'
        self.gamma_shape \
            = self.prior_estimate ** 2 / self.spread ** 2
        self.gamma_scale \
            = self.spread ** 2 / self.prior_estimate
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] = self.gamma_pdf(self.xvalues[i])
            self.prior_cdf[i] = self.gamma_cdf(self.xvalues[i])

    def gamma_pdf(self, xvalue):
        gamma_pdf = gamma.pdf(xvalue, self.gamma_shape, 0,
                              self.gamma_scale)
        return gamma_pdf

    def gamma_cdf(self, xvalue):
        gamma_cdf = gamma.cdf(xvalue, self.gamma_shape, 0,
                              self.gamma_scale)
        return gamma_cdf
        
    def normal_unlimited(self):
        self.distribution_description = ('Normal distribution (not ' +
            'truncated)')
        self.lower_limit = 'No lower limit'
        self.upper_limit = 'No upper limit'
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] = self.normal_unlimited_pdf(self.xvalues[i])
            self.prior_cdf[i] = self.normal_unlimited_cdf(self.xvalues[i])

    def normal_unlimited_pdf(self, xvalue):
        normal_unlimited_pdf = norm.pdf(xvalue, self.prior_estimate, self.spread)
        return normal_unlimited_pdf

    def normal_unlimited_cdf(self, xvalue):
        normal_unlimited_cdf = norm.cdf(xvalue, self.prior_estimate, self.spread)
        return normal_unlimited_cdf

    def normal_positive(self):
        self.distribution_description = ('Normal distribution truncated ' +
            'at zero only')
        self.lower_limit = 0
        self.upper_limit = 'No upper limit'
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] = self.normal_positive_pdf(self.xvalues[i])
            self.prior_cdf[i] = self.normal_positive_cdf(self.xvalues[i])

    def normal_positive_pdf(self, xvalue):
        normal_positive_pdf \
            = truncnorm.pdf(xvalue, - self.prior_estimate / self.spread, 1e10,
                            loc = self.prior_estimate, scale = self.spread)
        return normal_positive_pdf

    def normal_positive_cdf(self, xvalue):
        normal_positive_cdf \
            = truncnorm.cdf(xvalue, - self.prior_estimate / self.spread, 1e10,
                            loc = self.prior_estimate, scale = self.spread)
        return normal_positive_cdf

    def normal_truncated(self):
        self.distribution_description = ('Normal distribution truncated at ' +
            'defined points')
        self.lower_limit = self.limits[0]
        self.upper_limit = self.limits[1]
        for i in range(len(self.xvalues)):
            self.prior_pdf[i] = self.normal_truncated_pdf(self.xvalues[i])
            self.prior_cdf[i] = self.normal_truncated_cdf(self.xvalues[i])

    def normal_truncated_pdf(self, xvalue):
        normal_truncated_pdf \
            = truncnorm.pdf(xvalue,
                            (self.lower_limit - self.prior_estimate) / self.spread,
                            (self.upper_limit - self.prior_estimate) / self.spread,
                            loc = self.prior_estimate, scale = self.spread)
        return normal_truncated_pdf

    def normal_truncated_cdf(self, xvalue):
        normal_truncated_cdf \
            = truncnorm.cdf(xvalue,
                            (self.lower_limit - self.prior_estimate) / self.spread,
                            (self.upper_limit - self.prior_estimate) / self.spread,
                            loc = self.prior_estimate, scale = self.spread)
        return normal_truncated_cdf

    def pdf(self, xvalue):
        if self.distribution == 'gamma':
            pdf = self.gamma_pdf(xvalue)
        elif self.distribution == 'beta_full_range':
            pdf = self.beta_full_range_pdf(xvalue)
        elif self.distribution == 'beta_symmetric_params2':
            pdf = self.beta_symmetric_params2_pdf(xvalue)
        elif self.distribution == 'normal_unlimited':
            pdf = self.normal_unlimited_pdf(xvalue)
        elif self.distribution == 'normal_positive':
            pdf = self.normal_positive_pdf(xvalue)
        elif self.distribution == 'normal_truncated':
            pdf = self.normal_truncated_pdf(xvalue)
        return pdf

    def graph_prior(self):
        self.calculate_prior()
        pyplot.plot(self.xvalues, self.prior_pdf, 'r-', label = 'PDF')
        pyplot.plot(self.xvalues, self.prior_cdf, 'b-', label = 'CDF')
        pyplot.xlim(0., self.x_max_forgraph)
        pyplot.xlabel('Parameter value')
        pyplot.ylabel('Probability density')
        pyplot.legend()
        module_dir = os.path.dirname(__file__)
        os.chdir(os.path.join(module_dir, '..', 'graphs'))
        pyplot.savefig((self.name + '.jpg'))
        pyplot.close()
