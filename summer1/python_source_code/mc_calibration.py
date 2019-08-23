import theano.tensor as tt
import matplotlib.pyplot as plt
from python_source_code.tb_model import build_working_tb_model
import pandas as pd
import pymc3 as pm
import datetime
import theano
import numpy as np
import logging
logger = logging.getLogger("pymc3")
logger.setLevel(logging.DEBUG)

_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.DEBUG)

theano.config.optimizer='None'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'



def unc_run(theta):
    """
        Specify what type of object will be passed and returned to the Op when it is
    """
    beta, cdr_adjustment, start_time = theta
    tb_model = build_working_tb_model(beta, cdr_adjustment, start_time)
    tb_model.run_model()
    output_df = pd.DataFrame(tb_model.outputs, columns=tb_model.compartment_names)
    output_df.insert(loc=0, column="times", value=tb_model.times)

    # for 2015 year = 200-5
    age_15 = output_df['susceptibleXage_15'][195] + output_df['early_latentXage_15'][195] + output_df['late_latentXage_15'][195] + output_df['infectiousXage_15'][195] + \
             output_df['recoveredXage_15'][195]

    # for 2016 year = 200-4
    age_6 = output_df['susceptibleXage_6'][196] + output_df['early_latentXage_6'][196] + output_df['late_latentXage_6'][196] + output_df['infectiousXage_15'][196] + \
            output_df['recoveredXage_6'][196]

    perc_ltbi_age6 = (1 - output_df['susceptibleXage_6'][196]) / age_6
    prop_prev_age15 = output_df['infectiousXage_15'][195] / age_15

    return  np.asarray([prop_prev_age15, perc_ltbi_age6])


# define a theano Op for our likelihood function
class LogLike(tt.Op):

   """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
   """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta,  self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood

def my_loglike(theta, data, sigma ):
    model = unc_run(theta)
    return -(0.5/sigma**2)*np.sum((data - model)**2)

def mcmc(prior_distribution, outputs_unc, step_method_param='tb_n_contact'):


    # data prevelance and ltbi target
    data = np.asarray([0.0056, 9.6])

    sigma = 1.
    # create theano Op
    logl = LogLike(my_loglike, data, sigma)

    # calculate prior
    with pm.Model() as unc_model:

        beta = pm.Uniform('beta', lower=2.0, upper=100.0)

        cdr_adjustment = pm.Beta('cdr_adjustment', alpha=0.7, beta=0.15)

        start_time = pm.Uniform('start_time', lower=1830., upper=1920.)

        print('-----------------------------')
        beta = tt.printing.Print('beta')(beta)
        cdr_adjustment = tt.printing.Print('cdr_adjustment')(cdr_adjustment)
        start_time = tt.printing.Print('start_time')(start_time)
        print('-----------------------------')

        # convert beta, cdr_adjustment and start_time to a tensor vector
        theta = tt.as_tensor_variable([beta, cdr_adjustment, start_time])

        # likelihood
        # DensityDist (using lamdba function to "call" the Op)
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

        step = pm.Metropolis()
        trace = pm.sample(30, step, tune=0, chains=1,   progressbar=False)
        pm.summary(trace)

        df = pm.trace_to_dataframe(trace)
        df.to_csv('trace.csv')
        pd.scatter_matrix(df[:], diagonal='kde');

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.plot(df.ix[:, 'beta'], linewidth=0.7)
        plt.subplot(132)
        plt.plot(df.ix[:, 'cdr_adjustment'], linewidth=0.7);
        plt.subplot(133)
        plt.plot(df.ix[:, 'start_time'], linewidth=0.7);


    plt.show()


if __name__ == "__main__":
    start_timer_run = datetime.datetime.now()
    prior_distribution \
        = {'tb_n_contact': {'dist': 'uniform', 'lower': 2., 'upper' : 100.},
           'cdr_adjustment': {'dist': 'beta', 'alpha': .7, 'beta':.15},
           'start_time': {'dist': 'uniform', 'lower': 1830., 'upper': 1920.}}


    target_distribution \
        = {'target_prev' : {'type': 'prev', 'year': 2015, 'age':15 },
           'target_ltbi' : {'type': 'ltbi', 'year': 2016, 'age':6 }}

    mcmc(prior_distribution, target_distribution)
    diff = datetime.datetime.now() - start_timer_run
    print(diff)




