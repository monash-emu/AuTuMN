from numpy import mean, array, log, exp, nonzero, ndim
from math import sqrt
from scipy.special import betaln
from scipy.stats import norm
from autumn.model import make_steps
#from autofit import extractdata, list2dict
from copy import deepcopy
#from updatedata import unnormalizeF

def lnlike(population, par_arr, pars, nmeta, npops, data, verbose=2):
    """
    lnlike() Version 2015 Nov 09 Madhura Killedar

    DESCRIPTION
    calculated log likelihood for the data (e.g. prevalence, mortality) for the given parameter array par_arr

    INPUT
    par_arr:   array of parameters values of length nmeta+npops, including both model parameters and assumed bias (constant with time, one for each population
    pars   :   array of parameter names    
    nmeta  :   number of model parameters being calibrated
    npops  :   number of populations
    data   :   data dictionary including time information. data[x][y][z] is a list for population number x, data y (i.e. 'prev'), and z is either 'years' or 'vals'
    verbose:   flag to decide how much information to spit out during run

    OUTPUT
    returns lnlike_total (scalar float): log likelihood using whichever dataset is not-commented-out :)
    """
    try:
        Flist    = par_arr[0:nmeta]
        bias_arr = par_arr[nmeta:nmeta+npops]
    except IndexError as e:
        print(par_arr)
        print(e)
    
    D = deepcopy(population)
    for i in range(len(pars)): 
        if pars[i] in D.params:
            D.set_param(pars[i],par_arr[i])
    D.integrate_scipy(make_steps(0, 50, 1))
    
    #death == number of HIV-related deaths 
    #newtreat == number of people initiating ART
    #numtest == number of HIV tests per year
    #numinfect == number of new HIV infections
    #dx == number of HIV diagnoses
    #prev == HIV prevalence

    """
    [death, newtreat, numtest, numinfect, dx] = [[dict()], [dict()], [dict()], [dict()], [dict()]]
    for base in [death, newtreat, numtest, numinfect, dx]:
        base[0]['data'] = dict()
        base[0]['model'] = dict()
        base[0]['model']['x'] = S['tvec']
        if base == death:
            base[0]['data']['x'], base[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['death'][0])
            base[0]['model']['y'] = S['death'].sum(axis=0)
        elif base == newtreat:
            base[0]['data']['x'], base[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['newtreat'][0])
            base[0]['model']['y'] = S['newtx1'].sum(axis=0) + S['newtx2'].sum(axis=0)
        elif base == numtest:
            base[0]['data']['x'], base[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['numtest'][0])
            base[0]['model']['y'] = D['M']['hivtest'].sum(axis=0)*S['people'].sum(axis=0).sum(axis=0) #testing rate x population
        elif base == numinfect:
            base[0]['data']['x'], base[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['numinfect'][0])
            base[0]['model']['y'] = S['inci'].sum(axis=0)
        elif base == dx:
            base[0]['data']['x'], base[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['numdiag'][0])
            base[0]['model']['y'] = S['dx'].sum(axis=0)
    """

    # Prevalence
    # D['G']['npops'] = npops
    prev = [dict() for p in range(npops)]
    
    for p in range(npops):
        prev[p]['data'] = dict()
        prev[p]['model'] = dict()
        prev[p]['data']['x'] = data[p]['prev']['years']
        prev[p]['data']['y'] = data[p]['prev']['vals'] # The first 0 is for "best"
        prev[p]['model']['x'] = D.times
        prev[p]['model']['y'] = D.fractions['active'] # This is prevalence

    # WARNING Ideally use upper and lower limits in order to estimate the sig_p for each pop and each year 
    # sigp_p_y = (logit(max)-logit(min))/(2*1.96) # see Eaton & Hallett 2014 Eqn S20
    sigp = 0.25 # rough estimate

    """
    This is for when the model can caluclate diagnoses, new infections and deaths by year    
    
    
    # Diagnoses
    dx = [dict()]
    dx[0]['data'] = dict()
    dx[0]['model'] = dict()
    dx[0]['model']['x'] = S['tvec']
    dx[0]['data']['x'], dx[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['numdiag'][0])
    dx[0]['model']['y'] = S['dx'].sum(axis=0)

    # New Infections (Incidence)
    numinfect = [dict()]
    numinfect[0]['data'] = dict()
    numinfect[0]['model'] = dict()
    numinfect[0]['model']['x'] = S['tvec']
    numinfect[0]['data']['x'], numinfect[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['numinfect'][0])
    numinfect[0]['model']['y'] = S['inci'].sum(axis=0)

    # Number of TB-related deaths (Mortality)
    death = [dict()]
    death[0]['data'] = dict()
    death[0]['model'] = dict()
    death[0]['model']['x'] = S['tvec']
    death[0]['data']['x'], death[0]['data']['y'] = extractdata(D['G']['datayears'], D['data']['opt']['death'][0])
    death[0]['model']['y'] = S['death'].sum(axis=0)
    """

    lnlike_total = 0.
    # Log-Likelihood for Prevalence
    for ind in range(len(prev)):
        for y,year in enumerate(prev[ind]['data']['x']):
            model_ind = findinds(D.times, year)
            if len(model_ind)>0:
                model_p = prev[ind]['model']['y'][model_ind]
                data_p  = prev[ind]['data']['y'][y]
                # Normal + Cauchy
                #stat_epi = sterr(data_p) - sterr(model_p)
                stat_epi = logit(data_p) - logit(model_p)
                b=bias_arr[ind]
                lnlike_this = norm(b,sigp).logpdf(stat_epi)
                lnlike_total += lnlike_this
            else:
                print("no year in model_ind")
    
    
    """    
    # Log-Likelihood for Diagnoses
    for ind in range(len(dx)):
        for y,year in enumerate(dx[ind]['data']['x']):
            model_ind = findinds(S['tvec'], year)
            if len(model_ind)>0:
                model_dg = dx[ind]['model']['y'][model_ind]
                data_dg  = dx[ind]['data']['y'][y]
                lnlike_this = lnpoisson_rel(data_dg,model_dg)
                lnlike_total += lnlike_this
            else:
                print("no year in model_ind")
    # Log-Likelihood for Incidence
    for ind in range(len(numinfect)):
        for y,year in enumerate(numinfect[ind]['data']['x']):
            model_ind = findinds(S['tvec'], year)
            if len(model_ind)>0:
                model_inc = numinfect[ind]['model']['y'][model_ind]
                data_inc  = numinfect[ind]['data']['y'][y]
                lnlike_this = lnpoisson_rel(data_inc,model_inc)
                lnlike_total += lnlike_this
            else:
                print("no year in model_ind")
    # Log-Likelihood for Mortality
    for ind in range(len(death)):
        for y,year in enumerate(death[ind]['data']['x']):
            model_ind = findinds(S['tvec'], year)
            if len(model_ind)>0:
                model_inc = death[ind]['model']['y'][model_ind]
                data_inc  = death[ind]['data']['y'][y]
                lnlike_this = lnpoisson_rel(data_inc,model_inc)
                lnlike_total += lnlike_this
            else:
                print("no year in model_ind")
    """

    # return 0. # posterior = prior
    return lnlike_total


# Natural log of Poisson Dist: Probability of k successes given lam probability for each trial
# discarding the factorial that isn't dependent on lam
# both k and lam are number of success in a unit time
def lnpoisson_rel(k,lam):
    return k*log(lam) - lam


# logit transformation
def logit(p):
    return log(p/(1.-p))


def sterr(p):
    return sqrt(p/(1.-p))
    
def unnormalizeF(normF, M, G, normalizeall=False):
    """ Convert from F values where everything is 1 to F values that can be real-world interpretable. """
    unnormF = deepcopy(normF)
    for p in range(G['npops']):
        unnormF['init'][p] *= M['prev'][p] # Multiply by initial prevalence
        if normalizeall: unnormF['popsize'][p] *= M['popsize'][p][0] # Multiply by initial population size
    if normalizeall: unnormF['dx'][2] += G['datayears'].mean()-1 # Add mean data year
    return unnormF


def findinds(val1, val2=None, eps=1e-6):
    """
    Little function to find matches even if two things aren't eactly equal (eg. 
    due to floats vs. ints). If one argument, find nonzero values. With two arguments,
    check for equality using eps. Returns a tuple of arrays if val1 is multidimensional,
    else returns an array.
    
    Examples:
        findinds(rand(10)<0.5) # e.g. array([2, 4, 5, 9])
        findinds([2,3,6,3], 6) # e.g. array([2])
    
    Version: 2014nov27 by cliffk
    """

    if val2==None: # Check for equality
        output = nonzero(val1) # If not, just check the truth condition
    else:
        output = nonzero(abs(array(val1)-val2)<eps) # If absolute difference between the two values is less than a certain amount
    if ndim(val1)==1: # Uni-dimensional
        output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
    return output

