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
from numpy import isfinite
from scipy.stats import beta, gamma, norm, truncnorm

import autumn.base
import autumn.model
import autumn.plotting as plotting
import autumn.curve


def make_gamma_dist(mean, std):
    loc = 0
    shape = mean ** 2 / std ** 2  
    scale = std ** 2 / mean
    return gamma(shape, loc, scale)

# Following function likely to be needed later as we have calibration inputs
# at multiple time points
def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def is_positive_definite(v):
    return isfinite(v) and v > 0.0


class ModelRunner():

    def __init__(self):
        self.model = autumn.model.ConsolidatedModel(0, 0, 0, False, False, False)
        self.model.make_times(1900, 2015.1, 1.)
        self.is_last_run_success = False
        self.param_props_list = [
            { 
                'init': 14.,
                'scale': 10.,
                'key': 'tb_n_contact',
                'short': 'n',
                'format': lambda v: "%-2.0f" % v
            },
            {
                'init': .12,
                'scale': 1,
                'key': 'tb_proportion_early_progression',
                'short': 'early_prog',
                'format': lambda v: "%.4f" % v
            },
            { 
                'init': 0.007,
                'scale': 5,
                'key': 'tb_rate_late_progression',
                'short': 'late_rate',
                'format': lambda v: "%.4f" % v
            },
            {
                'init': .8,
                'scale': 1,
                'key': 'program_proportion_detect',
                'short': 'detect',
                'format': lambda v: "%.3f" % v
            },
            {
                'init': 3.,
                'scale': 1,
                'key': 'tb_timeperiod_activeuntreated',
                'short': 'time_active',
                'format': lambda v: "%.3f" % v
            }
        ]

    def set_model_with_params(self, param_dict):
        n_set = 0
        for key in param_dict:
            if key in self.model.params:
                n_set += 1
                self.model.set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

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
        # self.model.integrate_explicit()
        try:
            self.model.integrate_explicit()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False

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

        year_2015 = indices(self.model.times, lambda x: x >= 2015.)[0]

        prev_2015 = self.model.get_var_soln("prevalence")[year_2015]
        inc_2015 = self.model.get_var_soln("incidence")[year_2015]
        mort_2015 = self.model.get_var_soln("mortality")[year_2015]
        latent_2015 = self.model.broad_fraction_soln["latent"][year_2015] * 1E2
        pop_2015 = self.model.get_var_soln("population")[year_2015]

        ln_prior = 0.0
        ln_prior += make_gamma_dist(15., 10.).logpdf(param_dict['tb_n_contact'])
        ln_prior += make_gamma_dist(0.12, 0.1).logpdf(param_dict['tb_proportion_early_progression'])
        ln_prior += make_gamma_dist(0.007, 0.005).logpdf(param_dict['tb_rate_late_progression'])
        ln_prior += make_gamma_dist(0.87, 0.2).logpdf(param_dict['program_proportion_detect'])
        ln_prior += make_gamma_dist(3., 2.).logpdf(param_dict['tb_timeperiod_activeuntreated'])

        ln_posterior = 0.0

        ln_posterior += norm(280, 20.).logpdf(inc_2015)
        ln_posterior += norm(30., 10.).logpdf(latent_2015)
        ln_posterior += norm(417., 20.).logpdf(prev_2015)
        ln_posterior += norm(10., 3.).logpdf(mort_2015)
        ln_posterior += norm(99E6, 5E6).logpdf(pop_2015)

        ln_overall = ln_prior + ln_posterior

        print_dict = collections.OrderedDict()
        for props in self.param_props_list:
            val = param_dict[props["key"]]
            print_dict[props["short"]] = props["format"](val)
        print_dict["="] = "=="
        print_dict["pop_2015"] = "{:<4}".format("{:.0f}M".format(pop_2015/1E6))
        print_dict["latent_2015"] = "{:<4.0f}".format(latent_2015)
        print_dict["prev_2015"] = "{:<4.0f}".format(prev_2015)
        print_dict["inc_2015"] = "{:<4.0f}".format(inc_2015)
        print_dict["mort_2015"] = "{:<4.0f}".format(mort_2015)
        print_dict["-lnprob"] = "{:7.0f}".format(-ln_overall)
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


out_dir = "calibration_graphs"
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



