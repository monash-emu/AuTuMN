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
from scipy.stats import beta
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import truncnorm
import matplotlib


class Parameter:
    """"  Initialises parameters with distributions prior to model runs """
    def __init__(self, parameter_name, parameter_type,
                 distribution, prior_estimate, spread, limits,
                 model_implementation):
        self.parameter_name = parameter_name
        self.parameter_type = parameter_type
        available_types = ['proportion',
                           'rate',
                           'sojourn time',
                           'multiplier']
        assert self.parameter_type in available_types
        self.distribution = distribution
        available_distributions = ['beta_symmetric_params2',
                                   'beta_full_range', 'gamma',
                                   'normal_unlimited', 'normal_positive',
                                   'normal_truncated']
        assert distribution in available_distributions, \
            'Distribution not available'
        self.prior_estimate = prior_estimate
        self.spread = spread
        self.limits = limits

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
            print(type(self.limits))
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
        matplotlib.pyplot.plot(self.xvalues, self.prior_pdf, 'r-')
        matplotlib.pyplot.plot(self.xvalues, self.prior_cdf, 'b-')
        matplotlib.pyplot.xlim(0., self.x_max_forgraph)
        matplotlib.pyplot.show()


class Evidence:
    """ Object to summarise evidence for use in parameter estimation """
    def __init__(self, parameter_name, point_estimate, evidence_name,
                 location_evidence, explanation_evidence, reference):
        self.name = parameter_name
        self.estimate = point_estimate
        self.location = location_evidence
        self.evidence_name = evidence_name
        self.reference = reference
        self.explanation = explanation_evidence

    def __str__(self):
        return self.name

    def goto_evidence_directory(self):
        pyfile_string = sys.argv[0]
        autumn_directory = pyfile_string[0: pyfile_string.rfind('/')]
        self.evidence_directory = autumn_directory + '/evidence/'
        os.chdir(self.evidence_directory)

    def open_pdf(self):
        self.goto_evidence_directory()
        os.startfile(self.location)

    def write_explanation_document(self):
        self.goto_evidence_directory()
        file = open(self.evidence_name + '.txt', 'w')
        file.write('TITLE: ' + self.evidence_name + '\n\n' +
                   'REFERENCE: ' + self.reference + '\n\n' +
                   'POINT ESTIMATE: ' + str(self.estimate) + '\n\n' +
                   'EXPLANATION: ' + self.explanation)
        file.close()
