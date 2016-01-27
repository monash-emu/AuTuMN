from numpy import isfinite, inf, log, zeros, ones, sum, random, empty, arange, amin, amax, concatenate, array, reshape, asarray
#from scipy.stats import cauchy
from emcee import EnsembleSampler
from autumn.likelihood import lnlike
#from pylab import shape, reshape, array
from corner import corner
from matplotlib import pyplot as plt

def mcmc(population, meta_prefit, npops, data, nwalkers=80, nsteps=500, nburn=100, storage="mcmc"):
    """
    MCMC Version 2015 Jan 17 Madhura Killedar
    Last update: 2015 Nov 20 Madhura Killedar

    implementation of D. Forman-Mackay's emcee for Bayesian  parameter fitting
    emcee is a python implementation of affine invariant MCMC ensemble sampler

    INPUT
    population:  model population group for mcmc
    meta_prefit: as a starting proposal for meta-parameters, taken from model baseline (or mean).
                 the walkers begin in a ball around this point and the send a chain to explore the parameter space
    nwalkers:    number of walkers/chains
    nsteps:      total number of steps taken by each chain
    nburn:       assumed burn-in phase
    storage:     path to folder in which to place output files

    OUTPUT
    samples: non-burn-in steps of all chains combined, used to actually sample the posterior

    ON-THE-FLY OUTPUT
    fn_diagnostics (ASCII ~1KB) : a log file for useful diagnostics of the MCMC run
    fn_chains      (PNG ~200KB) : plot of some of the chains/walkers in MCMC run for some of the parameters
    fn_corner      (PNG  ~1MB)  : corner plot for MCMC sample of all parameters
    """

# -------------------------------------------------

    # on-the-fly output filenames
    fn_diagnostics = "MCMCdiagnostics.log"
    fn_chains = "plot-mcmc-chain.png"
    fn_corner = "plot-mcmc-corner.png"

    logfile = storage+"/"+fn_diagnostics
    logf = open(logfile, "w")

    # SETUP NUMBER OF PARAMETERS, WALKERS, STEPS TO USE
    nmeta = len(meta_prefit)-npops  # number of metaparams only
    nparam = len(meta_prefit)  # actual number of parameters being calibrated (includes bias)
    if nwalkers < 2*nparam:  # will increase number of walkers/chains if needed
        nwalkers = 2*nparam
        print("\n I'm increasing number of walkers")
    logf.write("\n Number of walkers = "+str(nwalkers))
    logf.write("\n Number of steps per walker = "+str(nsteps))
    logf.write("\n ... of which the first "+str(nburn)+" are considered in the burn-in phase")
    steps_used = (nsteps-nburn)*nwalkers
    logf.write("\n Therefore, number of samples used = "+str(steps_used))

    # PROPOSAL DISTRIBUTION
    pars, proposal_center = [], []
    for i in meta_prefit.keys():  # Converting dictionary to ordered list
        pars.append(i)
        proposal_center.append(meta_prefit[i])
    prop_str = ""
    for i in meta_prefit:
        prop_str = prop_str+" ("+i+": "+str(meta_prefit[i])+")"
    prop_str = "\n Initial proposal for each parameter = "+prop_str
    print(prop_str)
    logf.write(prop_str)

    # starting point for all walkers in a ball around the suspected centre
    proposal_dist = [proposal_center + 1.e-2*random.randn(nparam) for i in range(nwalkers)]

    # RUN EMCEE
    print("\n *** running emcee ***")
    sampler = EnsembleSampler(nwalkers, nparam, lnprob, args=(population, pars, meta_prefit, nmeta, npops, data))
    sampler.run_mcmc(proposal_dist, nsteps)

    # POST EMCEE DIAGNOSTICS
    samples = sampler.chain[:, nburn:, :].reshape((-1, nparam))
    steps_used = samples.shape[0]
    logf.write("\n Number of samples used = "+str(steps_used))
    # Autocorrelation time
    auto_time = sampler.acor
    auto_str = ""
    for i in range(nparam):
        auto_str = auto_str+" "+str(auto_time[i])
    auto_str = "\n\n Autocorrelation time for each parameter = "+auto_str
    print(auto_str)
    logf.write(auto_str)
    # Acceptance fraction
    accept_frac = sampler.acceptance_fraction
    accf_str = "\n Acceptance fractions = "
    for i in range(nwalkers):
        accf_str = accf_str+" "+str(accept_frac[i])
    print(accf_str)
    logf.write(accf_str)

    # CHAIN PLOT OF MCMC
    figwalkers = plt.figure()
    npar_plt = min(4, nparam) 
    nwalk_plt = min(20, nwalkers)
    axes_plt = empty(npar_plt, dtype=object)
    step = arange(nsteps)
    for i in arange(npar_plt):
        axes_plt[i] = figwalkers.add_subplot(npar_plt, 1, i+1)
        axes_plt[i] = plt.gca()
    for i in arange(npar_plt):
        k = i
        for j in arange(nwalk_plt):  # or all nwalkers
            position_plt = sampler.chain[j, :, k]
            axes_plt[i].plot(step,position_plt, '-')
        label_plt = 'par '+str(i)
        axes_plt[i].set_xlabel(label_plt)
        axes_plt[i].set_ylim(amin(sampler.chain[:nwalk_plt-1, :, k]), amax(sampler.chain[:nwalk_plt-1, :, k]))
    figwalkers.savefig(storage+"/"+fn_chains, format="png")

    """
    # DEBUG: USE PRIOR DISTRIBUTION INSTEAD OF POSTERIOR/MCMC RESULT
    samples = empty([steps_used,nparam])
    for i in arange(nparam):
        samples[:,i] = random.lognormal(1.,1.,steps_used)
    """

    # CORNER PLOT OF POSTERIOR SAMPLE
    labels_mcmc = empty(nparam, dtype=object)
    for k in range(nparam):
        labels_mcmc[k] = pars[k]
    fig_mcmc_corner = corner(samples, labels=labels_mcmc, truths=proposal_center)
    fig_mcmc_corner.savefig(storage+"/"+fn_corner, format="png")

    logf.close()
    return samples

# log posterior probability of given parameter array x (up to some additive factor)
# returns scalar float

def lnprob(x, population, pars, meta_mu, nmeta, npops, data):
    lp = lnprior(x, pars, meta_mu, nmeta, npops)
    if not isfinite(lp):
        return -inf
    return lp+lnlike(population, x, pars, nmeta, npops, data, verbose=0)


# log prior for given parameter array x. Needs to be written
# returns scalar float

def lnprior(x, pars, meta_prefit, nmeta,npops):
    lnpri_meta = 0.    
    xmin = empty(nmeta+npops)  # set boundaries of prior (non-zero prior prob for parameters)
    xmax = empty(nmeta+npops)
    xmin = [0.05 * meta_prefit[pars[p]] for p in range(nmeta+npops)]  # metaparams can only be 5% estimate
    xmax = [2 * meta_prefit[pars[p]] for p in range(nmeta+npops)]  # metaparams can only be twice estimate
    for p in range(nmeta+npops):
        if 'bias' in pars[p]:
            xmin[p] = -100.  # bias
            xmax[p] = 100.
    isabove = x > xmin
    isbelow = x < xmax
    iswithinbounds = isabove*isbelow
    if iswithinbounds.all():
        # PRIOR ON METAPARAMS
        for i in range(len(pars)):
            if pars[i] in meta_prefit.keys():
                lnpri_meta += log((x[i]-meta_prefit[pars[i]])**2)/2.
        # PRIOR ON BIAS
        bias = x[nmeta:nmeta+npops]
        # lnpri_bias = lnCauchy(bias)  # cauchy prior is too peaky
        lnpri_bias = sum(-log(0.25+abs(bias)))  # modified Jeffreys prior (0.25 currently sigp, typical scatter)
        # PRIOR
        lnpri = lnpri_meta + lnpri_bias
        return lnpri
    else:
        return -inf
    

# sum of the log of the (lognormal) likelihood for each variable
# returns scalar float
def lnlognormal(x, mu, sig):
    lnx = log(x)
    # term_lns = log(sig) # this factor should be absorbed into normalisation...
    term_e = (lnx-mu)**2/2./(sig**2)
    # return sum(-lnx -term_lns -term_e)
    return sum(-lnx-term_e)

def lnCauchy(x):
    # standard Cauchy PDF (0,1)
    # return sum(cauchy.logpdf(x))
    return sum(-log(1.+x**2))    # ln Cauchy written by hand
