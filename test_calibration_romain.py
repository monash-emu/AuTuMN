# -*- coding: utf-8 -*-
import os
import glob
import copy

import numpy
from numpy import isfinite
from scipy.stats import norm,  uniform, beta

import autumn.base
import autumn.model
import autumn.curve
import autumn.plotting
import autumn.data_processing
import autumn.tool_kit
from autumn.tool_kit import indices

import datetime
from autumn.spreadsheet import read_input_data_xls

from scipy.optimize import minimize
import openpyxl as xl


def is_positive_definite(v):
    return isfinite(v) and v > 0.0


class ModelRunner:

    def __init__(self):
        #self.mode = 'calibration'
        self.mode = 'uncertainty'
        self.country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
        self.inputs = autumn.data_processing.Inputs(True)
        self.inputs.read_and_load_data()
        n_organs = self.inputs.model_constants['n_organs'] #[0]
        n_strains = self.inputs.model_constants['n_strains'] #[0]
        is_quality = self.inputs.model_constants['is_lowquality'] #[0]
        is_amplification = self.inputs.model_constants['is_amplification'] #[0]
        is_misassignment = self.inputs.model_constants['is_misassignment'] #[0]
        self.model = autumn.model.ConsolidatedModel(
            None,  # Scenario to run
            self.inputs)

        self.is_last_run_success = False
        self.param_props_list = [  # the parameters that we are fitting
            {
                'init': 19.0,
                'key': u'tb_n_contact',
                'format': lambda v: "%.4f" % v,
                'bounds': [3., 30.],
                'width_95_prior':1.0
            },
            {
                'init': 0.4,
                'key': u'program_prop_death_reporting',
                'short': 'prop_death_reported',
                'format': lambda v: "%.4f" % v,
                'bounds': [0.1, 0.9],
                'width_95_prior': 0.20

             }#,
            #  {
            #     'init': 1909.,
            #     'scale': 1930.,
            #     'key': u'start_time',
            #     'short': 'start_time',
            #     'format': lambda v: "%-2.0f" % v, # integer
            #     'prior': uniform(1800., 130.),  # uniform distrib on [1800, 1930]
            #     'bounds': [1800.,1930.],
            #     'width_95_prior': 10.
            # }

        ]

        self.calib_outputs = [  # the targeted outputs
            {
                'key': 'incidence',
                'output_weight': 1.0, # how much we want this output to be taken into account.
                'times': None,
                'values': None,
                'time_weights': {2014: 1.}, # all weigths are equal to 1 by default. Specify if different
                'posterior_sd': 2.
            },
            {
                'key': 'mortality',
                'output_weight': 1.0,  # how much we want this output to be taken into account
                'times': None,
                'values': None,
                'time_weights': {2014: 1.},
                'posterior_sd': 0.1
            }
        ]


        self.param_ranges_unc = [
            {
                'key': u'tb_n_contact',
                'bounds': [3., 30.],
                'distribution': 'beta'
            },
            {
                'key': u'program_prop_death_reporting',
                'bounds': [0.1, 0.6],
                'distribution': 'beta'
            }
        ]

        self.outputs_unc = [
            {
                'key': 'incidence',
                'posterior_sd': None
            },
            {
                'key': 'mortality',
                'posterior_width': 2.0
            }
        ]


        for key, value in self.inputs.model_constants.items():
            if type(value) == float:
                self.model.set_parameter(key, value)

        if self.mode == 'calibration':
            for props in self.param_props_list:
                self.model.set_parameter(props['key'], props['init'])

        self.data_to_fit = {}
        self.get_data_to_fit() # collect the data regarding incidence , mortality, etc. from the model object
        self.best_fit = {}
        self.nb_accepted = 0

    def get_data_to_fit(self):
        var_to_iterate = self.calib_outputs # for calibration
        if self.mode == 'uncertainty':
            var_to_iterate = self.outputs_unc

        for output in var_to_iterate:
            if (output['key']) == 'incidence':
                self.data_to_fit['incidence'] = self.model.inputs.original_data['tb']['e_inc_100k']
                self.data_to_fit['incidence_low'] = self.model.inputs.original_data['tb']['e_inc_100k_lo']
                self.data_to_fit['incidence_high'] = self.model.inputs.original_data['tb']['e_inc_100k_hi']
            elif (output['key']) == 'mortality':
                self.data_to_fit['mortality'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k']
                self.data_to_fit['mortality_low'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k_lo']
                self.data_to_fit['mortality_high'] = self.model.inputs.original_data['tb']['e_mort_exc_tbhiv_100k_hi']
            else:
                print "Warning: Calibrated output %s is not directly available from the data" % output['key']

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
            bounds = self.param_props_list[i]["bounds"]
            if (p < bounds[0]) or (p > bounds[1]):
                #print "Warning: parameter%d=%f is outside of the allowed bounds" % (i, p)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        # self.model.integrate_explicit()
        try:
            print self.model.params['tb_n_contact']
            #self.model.integrate_runge_kutta()
            self.model.integrate_explicit()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False

    def ln_overall(self, params):
        self.run_with_params(params)
        if not self.is_last_run_success:
            return -numpy.inf

        param_dict = self.convert_param_list_to_dict(params)

        # ******** Posterior distributions *********
        ln_posterior = 0.0
        for output in self.calib_outputs: # for each targeted output
            if output['key'] in self.data_to_fit.keys():
                for i_times in range(len(self.data_to_fit[output['key']])):
                    time = self.data_to_fit[output['key']].keys()[i_times]
                    target = self.data_to_fit[output['key']][time]
                    year = indices(self.model.times, lambda x: x >= time)[0]
                    model_output = self.model.get_var_soln(output["key"])[year]


                    time_weight = 1.0
                    if time in output['time_weights'].keys():
                        time_weight = output['time_weights'][time]
                    w = output["output_weight"] * time_weight
                    ln_posterior += w * norm(target, output["posterior_sd"]).logpdf(model_output)
            else:
                for i_times in range(len(output["times"])):
                    time = output["times"][i_times]
                    year = indices(self.model.times, lambda x: x >= time)[0]
                    target = output["values"][i_times]
                    model_output = self.model.get_var_soln(output["key"])[year]

                    time_weight = 1.0
                    if time in output['time_weights'].keys():
                        time_weight = output['time_weights'][time]
                    w = output["output_weight"] * time_weight
                    if w > 0.:
                        ln_posterior += w * norm(target, output["posterior_sd"]).logpdf(model_output)

        ln_overall = ln_posterior

        return ln_overall

    def dist_squares(self,params):
        self.run_with_params(params)
        if not self.is_last_run_success:
            return -numpy.inf

        param_dict = self.convert_param_list_to_dict(params)

        dist = 0.0
        for output in self.calib_outputs:  # for each targeted output
            if output['key'] in self.data_to_fit.keys():
                for i_times in range(len(self.data_to_fit[output['key']])):
                    time = self.data_to_fit[output['key']].keys()[i_times]
                    target = self.data_to_fit[output['key']][time]
                    year = indices(self.model.times, lambda x: x >= time)[0]
                    model_output = self.model.get_var_soln(output["key"])[year]
                    if target != 0:
                        scale = target
                    else:
                        scale = 1.0

                    time_weight = 1.0
                    if time in output['time_weights'].keys():
                        time_weight = output['time_weights'][time]
                    w = output["output_weight"] * time_weight
                    if w > 0:
                        dist += w*((target - model_output)/scale)**2

            else:
                for i_times in range(len(output["times"])):
                    time = output["times"][i_times]
                    year = indices(self.model.times, lambda x: x >= time)[0]
                    target = output["values"][i_times]
                    model_output = self.model.get_var_soln(output["key"])[year]

                    time_weight = 1.0
                    if time in output['time_weights'].keys():
                        time_weight = output['time_weights'][time]
                    w = output["output_weight"] * time_weight
                    if w > 0.:
                        dist += w*(target - model_output)**2

        return dist

    def get_init_params(self):
        return [props['init'] for props in self.param_props_list]

    def maximize_ln_overall(self):

        def fun(pars):
            print('*******')
            print(pars)
            i = 0
            for props in self.param_props_list:
                pars[i] = pars[i] * (props["bounds"][1] - props["bounds"][0]) + props["bounds"][0]
                i += 1
            print('########')
            print(pars)
            y = - self.ln_overall(pars)
            return(y)

        init_params = []
        bounds = []
        for props in self.param_props_list:
            init = (props["init"]-props["bounds"][0])/(props["bounds"][1]-props["bounds"][0])
            init_params.append(init)
            bnds = [0.0, 1.0]
            bounds.append(bnds)

        print(bounds)
        init_params=[0.5, 0.5, 0.5]
        m = minimize(fun = fun, x0 = init_params, bounds = bounds, options={'disp': True, 'maxiter': 1}, method='SLSQP')
        best_theta = m.x
        i = 0
        for props in self.param_props_list:
            best_theta[i] = best_theta[i] * props["scale"]
            i += 1

        return best_theta

    def mcmc_romain(self, n_mcmc_step=40, max_iter=500, dist = False):
        self.nb_accepted = n_mcmc_step

        def get_sd_from_width(width):
            return width/(2.0*1.96)

        def update_par(pars):
            # pars is the former position for the different parameters
            new_pars = []
            i = 0
            for props in self.param_props_list:
                sd = get_sd_from_width(props['width_95_prior'])
                random = -100.
                bounds = props['bounds']
                while random<bounds[0] or random>bounds[1]:
                    random = norm.rvs(loc=pars[i], scale=sd, size=1)
                new_pars.append(random)
                i +=1
            return(new_pars)

        pars = []
        for props in self.param_props_list:
            pars.append(props['init'])

        if not dist:
            f = self.ln_overall(pars)
        else:
            f = -self.dist_squares(pars)

        n_accepted = 1
        n_candidates = 1
        acc = 1
        n_consecutive_rej = 0
        pars_accepted = numpy.zeros((n_mcmc_step, len(self.param_props_list)))
        f_accepted = numpy.zeros((n_mcmc_step, 1))
        pars_accepted[0, :] = pars
        f_accepted[0] = f
        print ("Initial value for f:" + str(f))

        while n_accepted < n_mcmc_step and n_candidates < max_iter:
            n_candidates += 1

            print '%d accepted / %d candidates' % (n_accepted, n_candidates)
            if acc == 1:
                cpt = 0
                s = ""
                for props in self.param_props_list:
                    s += props['key'] + " = " + str(pars[cpt]) + "    "
                    cpt += 1
                s += "objective f: " + str(f)
                print s
            if n_consecutive_rej >= 5 and n_candidates >= 10: # the algorithm struggles to accept a new candidate.
                for props in self.param_props_list:
                    props['width_95_prior'] *= 0.5 # new candidates generated in a narrower interval
                n_consecutive_rej = 0


            new_pars = update_par(pars)

            if not dist:
                new_f = self.ln_overall(new_pars)
            else:
                new_f = -self.dist_squares(new_pars)
            acc = 0
            if new_f > f:
                acc = 1
            else:
                if not dist:
                    acc = numpy.random.binomial(n=1, p=numpy.exp(new_f-f))
                else:
                    acc = 0

            if acc == 1:
                f = new_f
                pars = new_pars
                n_accepted += 1
                n_consecutive_rej = 0

                pars_accepted[n_accepted-1, :] = pars
                f_accepted[n_accepted-1, 0] = f

            else:
                n_consecutive_rej += 1

        self.pars = pars_accepted
        self.f = f_accepted
        self.rate_accepted = float(n_accepted) / float(n_candidates)

        self.get_best_fit()

    def get_best_fit(self):
        f_max = max(self.f)
        ind_max = [i for i, j in enumerate(self.f) if j == f_max][0]

        i = 0
        for props in self.param_props_list:
            self.best_fit[props['key']] = self.pars[ind_max][i]
            i += 1

    def get_initial_population(self, targeted_year=2014, tol=1000.):
        """
        Calculates the initial population that leads to the targeted current population
        Args:
            targeted_year: the time at which the population size should match
            tol: tolerated error
        Returns:
            The calculated initial population that should be used
        """

        if self.is_last_run_success == False: # need to run an intyegration
            self.model.integrate_explicit()

        indice_year = indices(self.model.times, lambda x: x >= targeted_year)[0]

        targeted_final_pop = self.model.inputs.original_data['tb_dict'][u'e_pop_num'][targeted_year]

        initial_pop = self.model.params[u'susceptible_fully']
        final_pop = sum(self.model.soln_array[indice_year, :])

        while abs(final_pop-targeted_final_pop) > tol:
            initial_pop = initial_pop * targeted_final_pop / final_pop
            # New run
            self.model.initial_compartments['susceptible_fully'] = initial_pop
            self.model.initialise_compartments()
            self.model.integrate_explicit()
            final_pop = sum(self.model.soln_array[indice_year, :])

        return initial_pop

    def write_best_fit_into_file(self):

        out_dir = 'calibrations/xls'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        name = 'calibrated_params_' + self.country + '_' + str(self.nb_accepted) + 'runs'
        path = os.path.join(out_dir, name)
        path = path + ".xlsx"

        wb = xl.Workbook()
        sheet = wb.active
        sheet.title = 'Parameters'

        cells_names = ['A1', 'A2']
        cells_vals = ['B1', 'B2']

        i = 0
        for par in self.best_fit.keys():
            sheet[cells_names[i]] = par
            sheet[cells_vals[i]] = self.best_fit[par]
            i += 1

        wb.save(path)

def run_calibration(n_runs, calibrated_params, targeted_outputs, dt=None):

    """
    run the automatic calibration for a country

    Args:
        n_runs: number of accepted parameter sets that we want
        calibration_params: dictionary defining the parameters to adjust
        targeted_outputs: dictionary defining the outputs that we are targeting
        dt: step time for integration. If None, it will be automatically determined to get optimal calculation time

    Returns:
        Nothing but writes a spreadsheet with the fitted parameters in calibrations/xls/...
        Also produces a graph

    """
    # Start timer
    start_realtime = datetime.datetime.now()

    model_runner = ModelRunner()
    print model_runner.country
    model_runner.param_props_list = calibrated_params
    model_runner.calib_outputs = targeted_outputs

    if dt is None:
        print "******** Automatic determination of the step-time *********"
        dts = [2., 1., 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.02, 0.01, 0.005, 0.001 ]
        ok = 0
        i = 0
        while ok == 0:
            val = dts[i]
            model_runner.model.time_step = val
            ok = 1
            try:
                model_runner.model.integrate_explicit()
            except:
                print "dt=" + str(val) + " fails"
                ok = 0
                i += 1
        dt = val
        print "********   dt=" + str(val) + " succeeds     ******** "

    model_runner.model.time_step = dt

    print('')
    print "******** Start MCMC simulation *********"
    model_runner.mcmc_romain(n_mcmc_step=n_runs, max_iter=1000, dist=True)
    model_runner.write_best_fit_into_file()

    out_dir = 'calibrations'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    name = 'calibrated_outputs_' + model_runner.country + '_' + str(model_runner.nb_accepted) + 'runs'
    base = os.path.join(out_dir, name)
    autumn.plotting.plot_outputs_against_gtb(
        model_runner.model, ["incidence", "mortality", "prevalence", "notifications"],
        model_runner.inputs.model_constants['recent_time'],
        'current_time',
        base + '.png',
        model_runner.country,
        scenario=None,
        figure_number=1)
    pngs = glob.glob(os.path.join(out_dir, '*png'))
    autumn.plotting.open_pngs(pngs)
    model_runner.write_best_fit_into_file()

    print("Time elapsed in running script is " + str(datetime.datetime.now() - start_realtime))

def run_uncertainty(n_runs, param_ranges_unc, outputs_unc, burn_in=10, dt=None, adaptive_search=True ,search_width=0.2):
    """
        run the uncertainty analysis for a country

        Args:
            n_runs: number of accepted parameter sets that we want
            param_ranges_unc: dictionary defining the parameter ranges
            outputs_unc: dictionary defining the outputs that we are targeting
            dt: step time for integration. If None, it will be automatically determined to get optimal calculation time
            adaptive_search: if True, the next candidate is generated from Normal distribution centred around current position
                             if False, the prior distribution is used to generate the parameter
            search_width: relevant when adaptive_search is True. Define the relative width of the 95% SI corresponding to the
                          new candidate generation from the normal distribution. It is relative to the width of the
                          attribute 'bounds' of 'param_ranges_unc'. i.e. search_width = 1.0 -> bounds is the 95% SI
        Returns:
            master storage unit that will keep track of all accepted parameter sets and associated model objects (with integration run)
        """
    model_shelf = []      # the master storage unit be returned

    model_runner = ModelRunner()
    print model_runner.country
    print "Uncertainty analysis"
    model_runner.param_ranges_unc = param_ranges_unc
    model_runner.outputs_unc = outputs_unc

    if dt is None:
        print "******** Automatic determination of the step-time *********"
        dts = [2., 1., 0.75, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.02, 0.01, 0.005, 0.001 ]
        ok = 0
        i = 0
        while ok == 0:
            val = dts[i]
            model_runner.model.time_step = val
            ok = 1
            try:
                model_runner.model.integrate_explicit()
            except:
                print "dt=" + str(val) + " fails"
                ok = 0
                i += 1
        dt = val
        print "********   dt=" + str(val) + " succeeds     ******** "

    model_runner.model.time_step = dt

    # Parameters candidates generation
    def generate_candidates(nb_candidates, param_ranges_unc):
        par_candidates = {}  # will store the candidates value
        for par_dic in param_ranges_unc:
            bound_low, bound_high = par_dic['bounds'][0], par_dic['bounds'][1]
            if par_dic['distribution'] == 'beta':
                x = numpy.random.beta(2.0, 2.0, nb_candidates)
                x = bound_low + x*(bound_high - bound_low)
            elif par_dic['distribution'] == 'uniform':
                x = numpy.random.uniform(bound_low,bound_high, nb_candidates)

            par_candidates[par_dic['key']] = x

        return par_candidates

    def update_par(pars):
        # pars is the former position for the different parameters
        new_pars = []
        i = 0
        for par_dict in param_ranges_unc:
            bounds = par_dict['bounds']
            sd = search_width*(bounds[1] - bounds[0]) / (2.0 * 1.96)
            random = -100.
            while random < bounds[0] or random > bounds[1]:
                random = norm.rvs(loc=pars[i], scale=sd, size=1)
            new_pars.append(random)
            i += 1
        return (new_pars)

    if not adaptive_search:
        nb_candidates = n_runs * 10
    else:
        nb_candidates = 1

    par_candidates = generate_candidates(nb_candidates=nb_candidates, param_ranges_unc=param_ranges_unc)

    # define the characteristics of the normal distribution for model outputs (incidence, mortality)
    def get_normal_char(model_runner, outputs_unc):
        normal_char = {} # store the characteristics of the normal distributions
        for output_dict in outputs_unc:
            normal_char[output_dict['key']] = {}
            if output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2.0 * 1.96)
                for year in model_runner.data_to_fit[output_dict['key']].keys():
                    mu = model_runner.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

            elif output_dict['key'] == 'incidence':
                for year in model_runner.data_to_fit[output_dict['key']].keys():
                    low = model_runner.data_to_fit['incidence_low'][year]
                    high = model_runner.data_to_fit['incidence_high'][year]
                    sd = (high - low) / (2.0 * 1.96)
                    mu = 0.5*(high + low)
                    normal_char[output_dict['key']][year] = [mu, sd]

        return normal_char

    normal_char = get_normal_char(model_runner=model_runner, outputs_unc=outputs_unc)

    # start simulation
    par_accepted = {}
    for par_dict in param_ranges_unc:
        par_accepted[par_dict['key']] = []
    n_accepted = 0
    i_candidates = 0
    j = 0
    prev_log_likelihood = -1e10
    params = []
    while n_accepted < n_runs + burn_in:
        new_params = []
        if not adaptive_search:
            for par_dict in param_ranges_unc:
                new_params.append(par_candidates[par_dict['key']][j])
        else:
            if i_candidates == 0:
                new_params = []
                for par_dict in param_ranges_unc:
                    new_params.append(par_candidates[par_dict['key']][j])
                    params.append(par_candidates[par_dict['key']][j])
            else:
                new_params = update_par(params)

        model_runner.run_with_params(new_params)
        if not model_runner.is_last_run_success:
            accepted = 0
        else:
            prior_log_likelihood = 0.0
            k = 0
            for par_dict in param_ranges_unc:
                par_val = new_params[k]
                # calculate the density of par_val
                bound_low, bound_high = par_dict['bounds'][0], par_dict['bounds'][1]
                if par_dict['distribution'] == 'beta':
                    #x = numpy.random.beta(2.0, 2.0, nb_candidates)
                    x = (par_val - bound_low)/(bound_high - bound_low)
                    prior_log_likelihood += beta.logpdf(x, 2.0, 2.0)
                elif par_dict['distribution'] == 'uniform':
                    prior_log_likelihood += numpy.log(1.0 / (bound_high - bound_low))

                k += 1

            posterior_log_likelihood = 0.0
            for output_dict in outputs_unc:
                dic = normal_char[output_dict['key']]
                for year in dic.keys():
                    year_indice = indices(model_runner.model.times, lambda x: x >= year)[0]
                    y = model_runner.model.get_var_soln(output_dict['key'])[year_indice]
                    mu, sd = dic[year][0], dic[year][1]
                    posterior_log_likelihood += norm.logpdf(y, mu, sd)

            log_likelihood = prior_log_likelihood + posterior_log_likelihood

            if log_likelihood >= prev_log_likelihood:
                accepted = 1
            else:
                accepted = numpy.random.binomial(n=1, p=numpy.exp(log_likelihood - prev_log_likelihood))

            if accepted == 1:
                n_accepted += 1
                k = 0
                for par_dict in param_ranges_unc:
                    par_accepted[par_dict['key']].append(new_params[k])
                    k += 1
                prev_log_likelihood = log_likelihood
                params = new_params

                if n_accepted > burn_in:
                    # model storage
                    params_dict = {}
                    k = 0
                    for par_dict in param_ranges_unc:
                        params_dict[par_dict['key']] = new_params[k]
                        k += 1

                    model_copy = copy.copy(model_runner.model)
                    model_shelf.append(
                        {
                            'model': model_copy,
                            'params': params_dict,
                            'log_likelihood': log_likelihood
                        }
                    )

        i_candidates += 1
        j += 1
        if j >= len(par_candidates.keys()) and not adaptive_search: # we need to generate more candidates
            par_candidates = generate_candidates(nb_candidates=nb_candidates, param_ranges_unc=param_ranges_unc)
            j = 0
        print (str(n_accepted) + ' accepted / ' + str(i_candidates) + ' candidates' )

    return model_shelf

def spaghetti_plot_uncertainty(model_shelf, country):
    """

    Args:
        model_shelf: master storage unit created from the function run_uncertainty(...)

    Returns:
        nothing. Just plot
    """
    n_runs = len(model_shelf)

    out_dir = 'uncertainty'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.join(out_dir, country + '_uncertainty_outputs_gtb')

    cpt = 0
    for model_dict in model_shelf:
        cpt += 1
        if (cpt == 1):
            sc = None
        else:
            sc = 2
        final_run = False
        if cpt == n_runs:
            final_run = True

        autumn.plotting.plot_outputs_against_gtb(
            model_dict['model'], ['incidence', 'mortality', 'prevalence', 'notifications'],
            1990.0,
            png= base + '.png',
            country=country,
            final_run=final_run,
            scenario=sc
            )


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #         Define and run the calibration from here          #   #

calibration_params = [  # the parameters that we are fitting
    {
        'key': u'tb_n_contact',
        'init': 7.,  # initial guess
        'bounds': [3., 30.],  # no parameter values will be generated outside of these bounds
        'width_95_prior': 1.0  # width of the interval containing 95% of the generated parameter values
    },
    {
        'key': u'program_prop_death_reporting',
        'init': 0.3,
        'short': 'prop_death_reported',
        'bounds': [0.1, 0.9],
        'width_95_prior': 0.20
    }
]
targeted_outputs = [  # the targeted outputs
    {
        'key': 'incidence',
        'output_weight': 1.0,  # how much we want this output to be taken into account.
        'times': None,  # only used when data is not available in GTB
        'values': None,  # only used when data is not available in GTB
        'time_weights': {},  # all weigths are equal to 1 by default. Specify if different (e.g {2014: 10., 1990: 10.}   )
        'posterior_sd': 2.
    },
    {
        'key': 'mortality',
        'output_weight': 1.0,  # how much we want this output to be taken into account
        'times': None,
        'values': None,
        'time_weights': {},
        'posterior_sd': 0.1
    }
]
n_runs = 5

#run_calibration(n_runs, calibration_params, targeted_outputs)





#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #         Define and run the the uncertainty from here          #   #
param_ranges_unc = [
    {
        'key': u'tb_n_contact',
        'bounds': [6.0, 6.2],
        'distribution': 'uniform'
    }#,
    # {
    #     'key': u'program_prop_death_reporting',
    #     'bounds': [0.1, 0.6],
    #     'distribution': 'beta'
    # }
]

outputs_unc = [
    {
        'key': 'incidence',
        'posterior_width': None
    }#,
    # {
    #     'key': 'mortality',
    #     'posterior_width': 1.0
    # }
]

n_runs = 5

model_shelf = run_uncertainty(n_runs, param_ranges_unc, outputs_unc, burn_in=5, dt=0.1, adaptive_search=True, search_width=0.2)
# Notes:
# adapt for time variant params
# prevent the algorithm from getting stuck
# include economics bit after integration of economics with the epi model


spaghetti_plot_uncertainty(model_shelf, 'Fiji')
print 'end'

#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
#   #  Determine the initial population size (uncomment code below))    #

# model_runner = ModelRunner()
# init_pop = model_runner.get_initial_population()
# print ('******* Initial population *********')
# print(init_pop)

