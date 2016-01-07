from matplotlib.pylab import zeros
from sim import Sim

class SimParameter(Sim):
    def __init__(self,meta,pars,name='Default'):
        # NB argument list will change once Project/Region object is created
        Sim.__init__(self,meta,pars,name)
        self.parameter_overrides = []

    def create_override(self,parname,startyear,endyear,startval,endval):
        # Register a parameter override
        override = {}
        override['parname'] = parname
        override['startval'] = startval
        override['startyear'] = startyear
        override['endval'] = endval
        override['endyear'] = endyear
        self.parameter_overrides.append(override)

    def makemodelpars(self):
        from numpy import linspace, ndim
        from utils import findinds
        
        Sim.makemodelpars(self)

        # Now compute the overrides as per scenarios.py -> makescenarios()
        for override in self.parameter_overrides:
            x = self.meta['tvec']
            y = self.parsmodel[override['parname']]
            initialindex = findinds(x, override['startyear'])
            finalindex = findinds(x, override['endyear'])

            # Todo: Error handling if initialindex or finalindex are not found goes here
            
            # Set new values
            npts = finalindex-initialindex
            newvalues = linspace(override['startval'], override['endval'], npts)
            y[initialindex:finalindex] = newvalues
            y[finalindex:] = newvalues[-1] # Fill in the rest of the array with the last value
            self.parsmodel[override['parname']] = y



