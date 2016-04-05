# -*- coding: utf-8 -*-

"""
Tests out the Parameter object in parameter_estimation.py (which is currently
just a prior setting object) and the Evidence object (which isn't yet used
within the Parameter object).
@author: James
"""

import os
import collections
import shutil

import numpy
import pylab
import emcee
import scipy.optimize as optimize
from numpy import isfinite
from scipy.stats import beta, gamma, norm, truncnorm
import math

import autumn.base
import autumn.model
import autumn.plotting as plotting
import autumn.curve


def make_gamma_dist(mean, std):
    loc = 0
    shape = mean ** 2 / std ** 2  
    scale = std ** 2 / mean
    return gamma(shape, loc, scale)


def is_positive_definite(v):
    return isfinite(v) and v > 0.0


class ModelRunner():

    def __init__(self):
        self.model = autumn.base.SimpleModel()
        self.model.make_times(1900, 2015, 1.)
        self.model.set_compartment(
            "susceptible", 50E6)
        self.is_last_run_success = False
        self.param_props_list = [
            { 
                'init': 30,
                'scale': 10.,
                'key': 'tb_n_contact',
                'short': 'n',
                'format': lambda v: "%-2.0f" % v
            },
            # { 
            #     'init': 50E6,
            #     'scale': 10E6,
            #     'key': 'init_population',
            #     'short': 'pop0',
            #     'format': lambda v: "%3.0fM" % (v/1E6)
            # },
            # { 
            #     'init': 0.5,
            #     'scale': 5,
            #     'key': 'tb_bcg_multiplier',
            #     'short': 'bcg',
            #     'format': lambda v: "%.4f" % v
            # },
            { 
                'init': .18,
                'scale': 1,
                'key': 'tb_rate_earlyprogress',
                'short': 'early',
                'format': lambda v: "%.4f" % v
            },
            { 
                'init': 1.8,
                'scale': 5,
                'key': 'tb_rate_stabilise',
                'short': 'stab',
                'format': lambda v: "%.4f" % v
            },
            { 
                'init': .2,
                'scale': 5,
                'key': 'tb_rate_recover',
                'short': 'recov',
                'format': lambda v: "%.4f" % v
            },
            { 
                'init': 0.001,
                'scale': 5,
                'key': 'tb_rate_lateprogress',
                'short': 'late',
                'format': lambda v: "%.4f" % v
            },
            { 
                'init': .01,
                'scale': 5,
                'key': 'tb_rate_death',
                'short': 'death',
                'format': lambda v: "%.4f" % v
            },
            # { 
            #     'init': .9,
            #     'scale': 1,
            #     'key': 'program_rate_detect',
            #     'short': 'detect',
            #     'format': lambda v: "%.4f" % v
            # },
        ]

    def set_model_with_params(self, param_dict):
        n_set = 0
        for key in param_dict:
            if key in self.model.params:
                n_set += 1
                self.model.set_param(key, param_dict[key])
        assert n_set == len(param_dict), "Not all params apply to model"

    def convert_param_list_to_dict(self, params):
        param_dict = {}
        for val, props in zip(params, self.param_props_list):
            param_dict[props['key']] = val
        return param_dict

    def run_with_params(self, params):
        for i, p in enumerate(params):
            if not is_positive_definite(p):
                print "Warning: parameter%d=%f is invalid for model" % (i, p)
                self.is_last_run_success = False
                return
        param_dict = self.convert_param_list_to_dict(params)
        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        self.model.integrate_explicit()
        # try:
        #     self.model.integrate_explicit()
        # except:
        #     print "Warning: parameters=%s failed with model" % params
        #     self.is_last_run_success = False

    def get_fraction_between_times(self, label, time0, time1):
        fraction = self.model.fraction_soln[label]
        times = self.model.times

        def fraction_at_time(test_t):
            best_i = 0
            best_diff = abs(times[best_i] - test_t)
            for i, t in enumerate(times):
                diff = abs(t - test_t)
                if diff < best_diff:
                    best_diff = diff
                    best_i = i
            return fraction[best_i]

        return abs(fraction_at_time(time0) - fraction_at_time(time1))

    def ln_overall(self, params):
        self.run_with_params(params)
        if not self.is_last_run_success:
            return -numpy.inf

        param_dict = self.convert_param_list_to_dict(params)

        final_pop = self.model.vars["population"]
        prevalence = self.model.vars["prevalence"]
        incidence = self.model.vars["incidence"]
        mortality = self.model.vars["mortality"]
        latent = (
            self.model.compartments["latent_late"]
             / self.model.vars["population"]
             * 1E5)

        fr_at_equil = self.get_fraction_between_times('latent_late', 1945, 1950)

        ln_prior = 0.0
        ln_prior += make_gamma_dist(40, 20).logpdf(param_dict['tb_n_contact'])

        ln_posterior = 0.0
        ln_posterior += - 1E8 * max(fr_at_equil - 0.05, 0.0)
        ln_posterior += norm(99E6, 5E6).logpdf(final_pop)
        ln_posterior += norm(30000, 100).logpdf(latent)
        ln_posterior += norm(417, 5).logpdf(prevalence)
        # ln_posterior += norm(280, 0.01).logpdf(incidence)
        # ln_posterior += norm(10, 1).logpdf(mortality)

        ln_overall = ln_prior + ln_posterior

        print_dict = collections.OrderedDict()
        for props in self.param_props_list:
            val = param_dict[props["key"]]
            print_dict[props["short"]] = props["format"](val)
        print_dict["="] = "=="
        print_dict["pop1"] = "{:<4}".format("{:.0f}M".format(final_pop/1E6))
        print_dict["latent"] = "{:2.0f}%".format(latent/1E5*100.)
        print_dict["prev"] = "{:<4.0f}".format(prevalence)
        print_dict["inci"] = "{:<4.0f}".format(incidence)
        print_dict["mort"] = "{:<4.0f}".format(mortality)
        print_dict["-lnprob"] = "{:7.0f}".format(-ln_overall)
        print_dict["fr_at_eq"] = "{:2.0f}%".format(fr_at_equil*100.0)
        print " ".join("%s=%s" % (k,v) for k,v in print_dict.items())

        return ln_overall

    def scale_params(self, params):
        return [
            param / d['scale']
            for param, d in
            zip(params, self.param_props_list)]

    def revert_scaled_params(self, scaled_params):
        return [
            scaled_param * prop['scale']
            for scaled_param, prop
            in zip(scaled_params, self.param_props_list)]

    def minimize(self):

        def scaled_min_fn(scaled_params):
            params = self.revert_scaled_params(scaled_params)
            return -self.ln_overall(params)
        
        n_param = len(self.param_props_list)
        init_params = [d['init'] for d in self.param_props_list]
        scaled_init_params = self.scale_params(init_params)

        # select the TNC method which is a basic method that
        # allows bouns in the search
        bnds = [(0, None) for i in range(n_param)]
        self.minimum = optimize.minimize(
            scaled_min_fn, scaled_init_params, 
            method='TNC', bounds=bnds)

        self.minimum.best_params = self.revert_scaled_params(self.minimum.x)
        return self.minimum

    def get_init_params(self):
        return [props['init'] for props in self.param_props_list]

    def mcmc(self, n_mcmc_step=40, n_walker_per_param=2):
        n_param = len(self.param_props_list)
        self.n_walker = n_walker_per_param * n_param
        self.n_mcmc_step = n_mcmc_step
        self.sampler = emcee.EnsembleSampler(
            self.n_walker, 
            n_param,
            lambda p: self.ln_overall(p))
        init_walkers = numpy.zeros((self.n_walker, n_param))
        init_params = self.get_init_params()
        for i_walker in range(self.n_walker):
             for i_param, init_x in enumerate(init_params):
                  init_walkers[i_walker, i_param] = init_x + 1e-1 * init_x * numpy.random.uniform()
        self.sampler.run_mcmc(init_walkers, self.n_mcmc_step)
        # Emma cautions here that if the step size is proportional to the parameter value,
        # then detailed balance will not be present.

    def plot_mcmc_params(self, base, n_burn_step=0):
        sampler = self.sampler
        n_walker, n_mcmc_step, n_param = sampler.chain.shape
        for i_param in range(n_param):
            pylab.clf()
            max_val = 0.0
            for i_walker in range(n_walker):
                vals = sampler.chain[i_walker, n_burn_step:, i_param]
                pylab.plot(range(len(vals)), vals)
                max_val = max(vals.max(), max_val)
            pylab.ylim([0, 1.2 * max_val])
            key = self.param_props_list[i_param]['key']
            plotting.set_axes_props(
                pylab.gca(), 
                'MCMC steps', 
                '', 
                "Parameter: " + key, 
                False)
            pylab.savefig("%s.param.%s.png" % (base, key))

    def gather_runs(self, n_burn_step=0, n_model_show=40):
        model = self.model
        sampler = self.sampler
        times = model.times

        chain = sampler.chain[:, n_burn_step:, :]
        n_param = chain.shape[-1]
        samples = chain.reshape((-1, n_param))
        self.average_params = numpy.average(samples, axis=0)

        self.runs = []
        n_sample = samples.shape[0]
        for i_sample in numpy.random.randint(n_sample, size=n_model_show):
            params = samples[i_sample, :]
            self.run_with_params(params)
            self.runs.append({
                'compartments': model.soln_array,
                'flow': model.flow_array,
                'fraction': model.fraction_array,
                'var': model.var_array,
            })

    def plot_var(self, label, base):
        times = self.model.times
        i_label = self.model.var_labels.index(label)
        pylab.clf()
        for i_run, run in enumerate(self.runs):
            vals = run['var'][:, i_label]
            pylab.plot(times, vals, color="k", alpha=0.1)
        self.run_with_params(self.average_params)
        pylab.plot(times, self.model.get_var_soln(label), color="r", alpha=0.8)
        plotting.set_axes_props(
            pylab.gca(), 
            'year', 
            label,
            'Modelled %s for selection of MCMC parameters' % label,
            False)
        pylab.savefig(base + '.var.' + label + '.png')

    def plot_fraction(self, label, base):
        times = self.model.times
        i_label = self.model.labels.index(label)
        pylab.clf()
        for i_run, run in enumerate(self.runs):
            vals = run['fraction'][:, i_label]
            pylab.plot(times, vals, color="k", alpha=0.1)
        self.run_with_params(self.average_params)
        pylab.plot(times, self.model.fraction_soln[label], color="r", alpha=0.8)
        plotting.set_axes_props(
            pylab.gca(),
            'year',
            label,
            'Modelled %s for selection of MCMC parameters' % label,
            False)
        pylab.savefig(base + '.fraction.' + label + '.png')

out_dir = "mcmc_graphs"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

model_runner = ModelRunner()

n_step = 40
base = os.path.join(out_dir, 'mcmc')
model_runner.mcmc(n_mcmc_step=n_step)
model_runner.model.make_graph(os.path.join(out_dir, 'workflow.png'))
model_runner.plot_mcmc_params(base)

n_burn_step = .5 * n_step
n_model = .5 * n_step
model_runner.gather_runs(n_burn_step=n_burn_step, n_model_show=n_model)
for var in ['population', 'incidence', 'prevalence', 'mortality']:
    model_runner.plot_var(var, base)
for label in ['latent_early', 'latent_late']:
    model_runner.plot_fraction(label, base)



