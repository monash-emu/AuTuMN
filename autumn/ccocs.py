# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:07:45 2015

@author: ntdoan

Make cost coverage and cost outcome curves. Mainly based on ccocs.py in Optima v2 branch, by Romesh 

"""
import abc
from math import log
import numpy as np
from copy import deepcopy
import pylab

class ccoc(object):
    # A ccoc object has
    # - A set of frontend parameters
    # - A selected functional form for the curve
    # - A mapping function that converts the FE parameters into BE parameters
    # - The ability to perturb BE parameters prior to retrieving the output value
    __metaclass__ = abc.ABCMeta

    def __init__(self,fe_params=None):
        if fe_params is None:
            fe_params = self.defaults()
        self.fe_params = fe_params
        self.is_linear = False # If function is linear, then ccoc.gradient() is mathematically valid

    @classmethod
    def fromdict(ccoc,ccocsdict):
        # This function instantiates the correct subtype based on simdict['type']
        ccoc_class = globals()[ccocsdict['type']] # This is the class corresponding to the CC form e.g. it could be  a <ccocs.cc_scaleup> object
        c = ccoc_class(None)
        c.load_dict(ccocsdict)
        return c

    def load_dict(self,ccocsdict):
        self.fe_params = ccocsdict['fe_params']

    def todict(self):
        ccocsdict = {}
        ccocsdict['type'] = self.__class__.__name__
        ccocsdict['fe_params'] = self.fe_params
        return deepcopy(ccocsdict)


    @abc.abstractmethod # This method must be defined by the derived class
    def function(self,x,p,t=None):
        # This function takes in either a spending value or coverage
        # and the functional parameters for the object, and returns
        # the coverage or the outcome
        # Optionally takes in an array of times as well. t should be the
        # same size as x
        pass

    @abc.abstractmethod # This method must be defined by the derived class
    def convertparams(self,perturb=False,bounds=None):
        # Take the current frontend parameters, and convert them into backend parameters
        # that can be passed into ccoc.function()
        # 'bounds' can be 'upper' or 'lower' to generate the curves displayed to the user
        # as the uncertainty ranges
        pass

    @abc.abstractmethod # This method must be defined by the derived class
    def defaults(self):
        # Return default fe_params for this CCOC
        pass

    def evaluate(self,x,t=None,perturb=False,bounds=None):
        p = self.convertparams(perturb,bounds)
        return self.function(x,p,t)

    def xlims(self):
        # Return sensible x-limits for the curve based on the parameters
        # This is used for plotting if no data is available
        return [0,1]

    def invert(self,y,t=None):
        p = self.convertparams()
        return self.inverse(y,p,t)

    def inverse(self,y,p,t=None):
        # This function should find the inverse numerically
        # but it can be overloaded by derived classes to provide
        # an analytic inverse
        raise Exception('Numerical inverse not implemented yet')

    def gradient(self,t=None,x=[0,1]):
        # Return the gradient of the linear CCOC, for each time
        if not self.is_linear:
            print ("WARNING: You have requested the gradient of a nonlinear CCOC")
        if t is None:
            y = self.evaluate(x,t)
            m = (y[1]-y[0])/(x[1]-x[0])
        else:
            m = np.zeros(t.shape)
            for i in xrange(0,len(m)):
                y = self.evaluate(np.array(x),np.array(t[i]))
                m[i] = (y[1]-y[0])/(x[1]-x[0])
        return m

######## SPECIFIC CCOC IMPLEMENTATIONS

class cc_scaleup(ccoc): #scale up 
    def function(self,x,p,t=None):
        return cceqn(x,p)
        
        """
        3-parameter equation defining cost-coverage curve.
        x is total cost, p is a list of parameters (of length 3):
        p[0] = saturation
        p[1] = inflection point
        p[2] = growth rate... 
        
        y = (p[3]-p[2]) * (2*p[0] / (1 + np.exp(-p[1]*x)) - p[0]) + p[2]
        
        Returns y which is coverage
        
        """

    def convertparams(self,perturb=False,bounds=None):
        
#        growthratel = np.exp((1-0.8)*log(1/0.2-1)+log(100))
#        growthratem = np.exp((1-0.8)*log(1/((0+1)/2)-1)+log(100))
#        growthrateu = np.exp((1-0.8)*log(1/0.2-1)+log(100))
#        convertedccparams = [[0.8, growthratem, 0.8],[0.8, growthratel, 0.8],[0.8, growthrateu, 1]] 
  
        growthratel = np.exp\
                        (\
                            (1.-self.fe_params['scaleup'])\
                            *log\
                                (\
                                self.fe_params['saturation']/self.fe_params['coveragelower']-1\
                                )\
                            +log(self.fe_params['funding'])\
                        )
        growthratem = np.exp\
                        (\
                            (1.-self.fe_params['scaleup'])\
                            *log\
                                (\
                                self.fe_params['saturation']\
                                /((self.fe_params['coveragelower']+self.fe_params['coverageupper'])/2)-1\
                                )\
                            +log(self.fe_params['funding'])\
                        )
        growthrateu = np.exp\
                        (\
                            (1.-self.fe_params['scaleup'])\
                            *log\
                                (\
                                self.fe_params['saturation']/self.fe_params['coverageupper']-1\
                                )\
                            +log(self.fe_params['funding'])\
                        )
        convertedccparams = [\
                                [self.fe_params['saturation'], growthratem, self.fe_params['scaleup']],\
                                [self.fe_params['saturation'], growthratel, self.fe_params['scaleup']],\
                                [self.fe_params['saturation'], growthrateu, self.fe_params['scaleup']]\
                            ]
                    
                         
                            
        if bounds==None: #bounds of coverage data 
            return convertedccparams[0]
        elif bounds=='lower':
            return convertedccparams[1]
        elif bounds=='upper':
            return convertedccparams[2]
        else:
            raise Exception('Unrecognized bounds')

    def defaults(self): #default values 
        fe_params = {}
        fe_params['scaleup'] = 1.
        fe_params['saturation'] = 1.
        fe_params['coveragelower'] = 0.
        fe_params['coverageupper'] = 1.
        fe_params['funding'] = 0.
        return fe_params

    def xlims(self):
        return [0,2e6]




#### This cost-outcome fuction did not get used in Robyn's and Romesh code 

class cco_scaleup(ccoc): #scale up 
    def function(self,x,p,t=None):
        return ccoeqn(x,p)
      
        '''  
        5-parameter equation defining cost-outcome curves.
        x is total cost, p is a list of parameters (of length 5):
        p[0] = saturation
        p[1] = inflection point
        p[2] = growth rate...
        p[3] = outcome at zero coverage
        p[4] = outcome at full coverage
        Returns y which is coverage.   
        y = (p[4]-p[3]) * (p[0] / (1 + np.exp((log(p[1])-np.log(x))/(1-p[2])))) + p[3]
        '''

    def convertparams(self,perturb=False,bounds=None):
  
        growthratel = np.exp(
            ( 1 - self.fe_params['scaleup'] ) \
            * log(
                self.fe_params['saturation']
                   / self.fe_params['coveragelower']
                - 1
              )
            + log( self.fe_params['funding'] )
        )
        growthratem = np.exp(
            (1-self.fe_params['scaleup'])
                * log(
                self.fe_params['saturation']/
                     ((self.fe_params['coveragelower']+self.fe_params['coverageupper'])/2)-1
                  )
            + log(self.fe_params['funding'])
        )
        growthratem = np.exp(
            ( 1 - self.fe_params['scaleup'] ) \
              * log(
                   self.fe_params['saturation']
                      /  ( ( self.fe_params['coveragelower'] + self.fe_params['coverageupper'] )
                            / 2 
                          )
                      - 1 
                )
            + log(self.fe_params['funding'])
            )
        growthrateu = np.exp\
                        (\
                            (1-self.fe_params['scaleup'])\
                            *log\
                                (\
                                self.fe_params['saturation']/self.fe_params['coverageupper']-1\
                                )\
                            +log(self.fe_params['funding'])\
                        )
        convertedccparams = [\
            [self.fe_params['saturation'], growthratem, self.fe_params['scaleup']],\
            [self.fe_params['saturation'], growthratel, self.fe_params['scaleup']],\
            [self.fe_params['saturation'], growthrateu, self.fe_params['scaleup']]\
        ]
                    
        if bounds==None: #bounds of coverage data 
            return convertedccparams[0]
        elif bounds=='lower':
            return convertedccparams[1]
        elif bounds=='upper':
            return convertedccparams[2]
        else:
            raise Exception('Unrecognized bounds')

    def defaults(self): #default values 
        fe_params = {}
        fe_params['scaleup'] = 1
        fe_params['saturation'] = 1
        fe_params['coveragelower'] = 0
        fe_params['coverageupper'] = 1
        fe_params['funding'] = 0
        fe_params['outcomezerocov'] = 0
        fe_params['outcomefullcov'] = 1
        return fe_params

    def xlims(self):
        return [0,2e6]


class cc_noscaleup(ccoc): # no scale up 
    def function(self,x,p,t=None):
        return cc2eqn(x,p)
        
        """
        2-parameter equation defining cc curves.
        x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
        
        y =  2*p[0] / (1 + np.exp(-p[1]*x)) - p[0]
        
        Returns y which is coverage
        
        """

    def convertparams(self,perturb=False,bounds='lower'):
        growthratel =   (-1./self.fe_params['funding'])\
                        *log\
                            (\
                                (2.*self.fe_params['saturation'])\
                                /(self.fe_params['coveragelower']+self.fe_params['saturation'])\
                                - 1.\
                            )
                            
        growthratem = \
            np.exp( -1. / self.fe_params['funding'] ) \
                 * log(\
                        ( 2. * self.fe_params['saturation'] )
                          / ( 0.5 * ( self.fe_params['coveragelower'] + self.fe_params['coverageupper'] ) \
                               + self.fe_params['saturation']\
                            )
                        -1.
                   )
                            
        print self.fe_params['funding'], growthratem
        numerator = ( 2. * self.fe_params['saturation'])                         
        denominator = (self.fe_params['coverageupper'] + self.fe_params['saturation'])
        constant = -1.
        x = numerator / denominator + constant
        growthrateu =   (-1./self.fe_params['funding'])\
                         * log(
                                x
                           )        
                                
        convertedccparams = [\
                                [self.fe_params['saturation'], growthratem],\
                                [self.fe_params['saturation'], growthratel],\
                                [self.fe_params['saturation'], growthrateu]\
                            ]
        if bounds==None:
            return convertedccparams[0] # [self.fe_params['saturation'], growthratem]
        elif bounds=='lower':
            return convertedccparams[1] # [self.fe_params['saturation'], growthratel]
        elif bounds=='upper':
            return convertedccparams[2] # [self.fe_params['saturation'], growthrateu]
        else:
            raise Exception('Unrecognized bounds')

    def defaults(self):
        fe_params = {}
        fe_params['saturation'] = 1.
        fe_params['coveragelower'] = 0.
        fe_params['coverageupper'] = 0.9
        fe_params['funding'] = 1000000
        return fe_params

    def xlims(self):
        return [0,2e6]




### This cost-outcome form did not get used in Robyn and Romesh code 

class cco_noscaleup(ccoc): # no scale up 
    def function(self,x,p,t=None):
        return cco2eqn(x,p)
        
        """
        4-parameter equation defining cost-outcome curves.
        x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
        p[2] = outcome at zero coverage
        p[3] = outcome at full coverage
        Returns y which is coverage.'''
        y = (p[3]-p[2]) * (2*p[0] / (1 + np.exp(-p[1]*x)) - p[0]) + p[2] 
        """

    def convertparams(self,perturb=False,bounds=None):
        growthratel =   (-1/self.fe_params['funding'])\
                        *log\
                            (\
                                (2*self.fe_params['saturation'])\
                                /(self.fe_params['coveragelower']+self.fe_params['saturation'])\
                                - 1\
                            )
                            
        growthratem =   (-1/self.fe_params['funding'])\
                        *log\
                            (\
                                (2*self.fe_params['saturation'])\
                                /(\
                                    ((self.fe_params['coveragelower']+self.fe_params['coverageupper'])/2)\
                                    +self.fe_params['saturation']\
                                )\
                            -1)  
                            
        growthrateu =   (-1/self.fe_params['funding'])\
                        *log\
                            (\
                                (2*self.fe_params['saturation'])\
                                /(self.fe_params['coverageupper']+self.fe_params['saturation'])\
                            -1)        
                                
        convertedccparams = [\
                                [self.fe_params['saturation'], growthratem],\
                                [self.fe_params['saturation'], growthratel],\
                                [self.fe_params['saturation'], growthrateu]\
                            ]
        if bounds==None:
            return convertedccparams[0] # [self.fe_params['saturation'], growthratem]
        elif bounds=='lower':
            return convertedccparams[1] # [self.fe_params['saturation'], growthratel]
        elif bounds=='upper':
            return convertedccparams[2] # [self.fe_params['saturation'], growthrateu]
        else:
            raise Exception('Unrecognized bounds')

    def defaults(self):
        fe_params = {}
        fe_params['saturation'] = 1
        fe_params['coveragelower'] = 0
        fe_params['coverageupper'] = 1
        fe_params['funding'] = 0
        fe_params['outcomezerocov'] = 0
        fe_params['outcomefullcov'] = 1
        return fe_params

    def xlims(self):
        return [0,2e6]



class co_cofun(ccoc):

    def __init__(self,fe_params=None):
        ccoc.__init__(self,fe_params)
        self.is_linear = True

    def function(self,x,p,t=None):
        return coeqn(x,p)

    def inverse(self,y,p,t=None):
        return (y-p[0])/(p[1]-p[0]) 
        
    def convertparams(self,perturb=False,bounds=None):
        # fe_params: list. Contains parameters for the coverage-outcome curves,
                     #obtained from the GUI
        #     fe_params[0] = the lower bound for the outcome when coverage = 0
        #     fe_params[1] = the upper bound for the outcome when coverage = 0
        #     fe_params[2] = the lower bound for the outcome when coverage = 1
        #     fe_params[3] = the upper bound for the outcome when coverage = 1
        
        # Mean and standard deviation calcs at zero coverage   
        # see below for why devided by six for std calculation
            ## http://www.statit.com/support/quality_practice_tips/estimating_std_dev.shtml
        
        muz, stdevz =   (self.fe_params[0]+self.fe_params[1])/2,\
                        (self.fe_params[1]-self.fe_params[0])/6 
                        
        # Mean and standard deviation calcs at full coverage                     
        muf, stdevf =   (self.fe_params[2]+self.fe_params[3])/2,\
                        (self.fe_params[3]-self.fe_params[2])/6 
       
        convertedcoparams = [muz, stdevz, muf, stdevf]
       
       # DEBUG CODE, DO THIS PROPERLY LATER        
        convertedcoparams = [\
                                [muz,muf],[muz-stdevz,muf-stdevf],\
                                [muz+stdevz,muf+stdevf]\
                            ] 
        if bounds==None:
            return convertedcoparams[0] # [muz,muf]
        elif bounds=='lower':
            return convertedcoparams[1] # [muz-stdevz,muf-stdevf]
        elif bounds=='upper':
            return convertedcoparams[2]  # [muz+stdevz,muf+stdevf]
        else:
            raise Exception('Unrecognized bounds')

    def defaults(self):
        return [0,0,1,1] # [zero coverage lower, zero coverage upper, full coverage lower, full coverage upper]

    def delta_out(self,t=None):
        grad = np.mean(self.fe_params[2:])-np.mean(self.fe_params[0:2])
        if t is not None:
            grad *= np.ones(t.shape)
        return grad

class co_linear(ccoc):

    def __init__(self,fe_params=None):
        ccoc.__init__(self,fe_params)
        self.is_linear = True

    def function(self,x,p,t=None):
        return linear(x,p)

    def convertparams(self,perturb=False,bounds=None):
        return self.fe_params

    def defaults(self):
        return [1,0] # [gradient intercept]

class linear_timevarying(ccoc):
    # A time-varying linear CCOC
    def __init__(self,fe_params=None):
        ccoc.__init__(self,fe_params)
        self.is_linear = True

    def function(self,x,p,t):
        # Linearly interpolate the fe_params gradient to get the current outcome gradient
        gradient_m = (p['unit_cost'][1]-p['unit_cost'][0])/(p['time'][1]-p['time'][0])
        gradient_b = p['unit_cost'][0]  -p['time'][0]*gradient_m
        current_gradient = t*gradient_m + gradient_b # This is the current unit cost (vector)
       
        # Linearly interpolate the fe_params intercept to get the current outcome intercept
        baseline_m = (p['baseline'][1]-p['baseline'][0])/(p['time'][1]-p['time'][0])
        baseline_b = p['baseline'][0]  -p['time'][0]*baseline_m
        current_baseline = t*baseline_m + baseline_b # This is the current zero_coverage outcome (vector)

        # What is the current parameter value?
        return current_gradient*x + current_baseline

    def convertparams(self,perturb=False,bounds=None):
        return self.fe_params

    def defaults(self):
        fe_params = dict()
        fe_params['time'] = [2015.0,2030.0] # t-values corresponding to unit cost
        fe_params['unit_cost'] = [1.0,2.0] # Unit cost - gradient of the CCOC
        fe_params['baseline'] = [0.0,0.0] # Parameter value with zero coverage
        return fe_params

class identity(ccoc):

    def __init__(self,fe_params=None):
        ccoc.__init__(self,fe_params)
        self.is_linear = True

    def function(self,x,p,t=None):
        return x

    def convertparams(self,perturb=False,bounds=None):
        return None

    def defaults(self):
        return None

class null(ccoc):
    def function(self,x,p,t=None):
        return None

    def convertparams(self,perturb=False,bounds=None):
        return None

    def defaults(self):
        return None
        
 

############## FUNCTIONAL FORMS COPIED FROM makeccocs.py ##############
def linear(x,params):
    return params[0]*x+params[1]

def cc2eqn(x, p):
    '''
    2-parameter equation defining cc curves.
    x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
    Returns y which is coverage. '''
    y =  2*p[0] / (1 + np.exp(-p[1]*x)) - p[0]
    return y
    
def cco2eqn(x, p):
    '''
    4-parameter equation defining cost-outcome curves.
    x is total cost, p is a list of parameters (of length 2):
        p[0] = saturation
        p[1] = growth rate
        p[2] = outcome at zero coverage
        p[3] = outcome at full coverage
    Returns y which is coverage.'''
    y = (p[3]-p[2]) * (2*p[0] / (1 + np.exp(-p[1]*x)) - p[0]) + p[2]
    return y

def cceqn(x, p, eps=1e-3):
    '''
    3-parameter equation defining cost-coverage curve.
    x is total cost, p is a list of parameters (of length 3):
        p[0] = saturation
        p[1] = inflection point
        p[2] = growth rate... 
    Returns y which is coverage.
    '''
    y = p[0] / (1 + np.exp((log(p[1])-np.log(x))/max(1-p[2],eps)))

    return y

def coeqn(x, p):
    '''
    Straight line equation defining coverage-outcome curve.
    x is coverage, p is a list of parameters (of length 2):
        p[0] = outcome at zero coverage
        p[1] = outcome at full coverage
    Returns y which is outcome.
    '''
    y = (p[1]-p[0]) * np.array(x) + p[0]

    return y
    
def ccoeqn(x, p):
    '''
    5-parameter equation defining cost-outcome curves.
    x is total cost, p is a list of parameters (of length 5):
        p[0] = saturation
        p[1] = inflection point
        p[2] = growth rate...
        p[3] = outcome at zero coverage
        p[4] = outcome at full coverage
    Returns y which is coverage.
    '''
    y = (p[4]-p[3]) * (p[0] / (1 + np.exp((log(p[1])-np.log(x))/(1-p[2])))) + p[3]

    return y
