from matplotlib.pylab import zeros, array
from copy import deepcopy


# Adding comment to check changes with github.

def model(sim):
    # -*- coding: utf-8 -*-
    """
    MODEL

    Implement the ODEs of the model.

    Dimensions are:
        - Disease state:
            - Susceptible
            - Latent
            - Active
            - Detected
            - Treated
        - HIV status
        - Age
        - Smear status
        - Retreatment

    Types of flow:
        - Birth (-> susceptible)
        - Infection (susceptible -> latent)
        - Stabilization (latent-A <-> latent-B)
        - Latent treatment (latent -> susceptible)
        - Progression (latent -> active)
        - Detection (active -> detected)
        - Recovery (active -> latent)
        - Treatment (detected -> treatment-infectious)
        - Default (treatment -> active)
        - Clearance (treatment-infectious -> treatment-noninfectious)
        - Success (treatment-noninfectious -> susceptible)
        - Susceptibility testing (treatment-S/M/X <-> treatment-M/X)
        - Death (all -> )
        

    Version: 2015aug06 by cliffk
    """

    meta = deepcopy(sim.meta)
    pars = deepcopy(sim.parsmodel)

    people = sim.initialconditions()
        
    iS, iL, iA, iD, iT = meta['iS'], meta['iL'], meta['iA'], meta['iD'], meta['iT']
    ninfections, nrecovered, nprogress, ndetected, ntreated, ntbdeaths = zeros(meta['npts']), zeros(meta['npts']), zeros(meta['npts']), zeros(meta['npts']), zeros(meta['npts']), zeros(meta['npts'])

    for t in xrange(meta['npts']):
        
        # Change in susceptibles
        nbirths = sum(people[:,t]) * pars['birth'][t]
        ninfections[t] = people[iS,t] * sum(people[[iA,iD],t]) / sum(people[:,t]) * pars['ncontacts'][t]
        nrecovered[t] = people[iT,t] * pars['recov'][t]
        dS = people[iS,t]*(-pars['death'][t]) + nbirths - ninfections[t] + nrecovered[t]
        
        # Change in latent
        nprogress[t] = people[iL,t] * pars['progress'][t]
        dL = people[iL,t]*(-pars['death'][t]) + ninfections[t] - nprogress[t]
        
        # Change in active
        ndetected[t] = people[iA,t] * pars['test'][t]
        activedeaths = people[iA,t] * pars['tbdeath'][t]
        ntbdeaths[t] += activedeaths
        dA = -activedeaths + nprogress[t] - ndetected[t]
        
        # Change in detected
        ntreated[t] = people[iD,t] * pars['treat'][t]
        detecteddeaths = people[iD,t] * pars['tbdeath'][t]
        ntbdeaths[t] += detecteddeaths
        dD = -detecteddeaths + ndetected[t] - ntreated[t]
        
        # Change in treated
        dT = people[iT,t]*(-pars['treatdeath'][t]) + ntreated[t] - nrecovered[t]
        
        # Update population array
        if t<meta['npts']-1: 
            people[:,t+1] = people[:,t] + meta['dt'] * array([dS, dL, dA, dD, dT])
        
            # Do error checking
            if any(people[:,t+1]<0):
                print('Negative people found')
                import traceback; traceback.print_exc(); import pdb; pdb.set_trace()
            
            if all(array([pars['birth'][t], pars['death'][t], pars['tbdeath'][t], pars['treatdeath'][t]])==0):
                if abs(sum(people[:,t+1])-sum(people[:,t])) > 1e-6:
                    print('Number of people is not consistent')
                    import traceback; traceback.print_exc(); import pdb; pdb.set_trace()

    output = dict()
    output['people'] = people
    output['ninfections'] = ninfections
    output['nrecovered'] = nrecovered
    output['nprogress'] = nprogress
    output['ndetected'] = ndetected
    output['ntreated'] = ntreated
    output['ntbdeaths'] = ntbdeaths

    return output