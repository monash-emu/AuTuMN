# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:07:45 2015

@author: ntdoan

Make cost coverage and cost outcome curve 

"""

from math import exp
import numpy as np
import pylab 

######## INPUT PARAMETERS ########

class input_parameters():

    def __init__(self):
        
        ''' for cost-coverage curve'''

        self.input_parameters("saturation", 0.82)
        self.input_parameters("coverage", 0.33)
        self.input_parameters("scale_up_factor", 0.5)
        self.input_parameters("unit_cost", 20)
        self.input_parameters("pop_size", 1e5)
        self.input_parameters("spending_lowerlimit", 0.01)
        self.input_parameters("spending_upperlimit", 1e6)
        self.input_parameters("spending_stepsize", 100)
        self.input_parameters("spending_reflection_point", 1e6)
        
        ''' for coverage-outcome curve ''' 
        ''' outcome here is rate of treatment completion ''' 
        
        self.input_parameters("outcome_zero_coverage", 0.1)
        self.input_parameters("outcome_full_coverage", 0.9)
        self.input_parameters("coverage_lowerlimit", 0.01)
        self.input_parameters("coverage_upperlimit", 1)
        self.input_parameters("coverage_stepsize", 1)
    
    def calculate_vars(self): 
        
        ''' for cost-coverage curve ''' 
        
        self.spending_num = (self.input_parameters["spending_uppperlimit"] -
                            self.input_parameters["spending_lowerlimit"])/\
                            self.input_parameters["spending_stepsize"]
                            
        self.spending = np.linspace(self.input_parameters["spending_lowerlimit"],\
                                    self.input_parameters["spending_upperlimit"],\
                                    self.spending_num) 
        
        ''' for coverage-outcome curve ''' 
        
        self.coverage_num = (self.input_parameters["coverage_uppperlimit"] -
                            self.input_parameters["coverage_lowerlimit"])/\
                            self.input_parameters["coverage_stepsize"]
                            
        self.coverage = np.linspace(self.input_parameters["coverage_lowerlimit"],\
                                    self.input_parameters["coverage_upperlimit"],\
                                    self.coverage_num) 
                                    
                                    
######## FUNCTIONAL FORMS ##########
        
class cost_coverage():

    def cost_coverage_curve(self): #Sigmoid shape 
        
        '''Returns coverage in a given year for a given spending amount.'''
        
        ''' Well-established programs, when unit cost is important''' 
        
        y = ((2*self.input_parameters["saturation"])/
                (1 + exp((-2*self.spending)/
                         (self.input_parameters["unit_cost"]*self.input_parameters["pop_size"]))
                )   
            ) - self.input_parameters["saturation"]
            
        return y 
        
        
        ''' Scale-up factor, no unit cost ''' 
        
        y = self.input_parameters["saturation"]/\
            (1 + (
                    ((self.input_parameters["saturations"]/self.input_parameters["coverage"]) - 1) *
                    (self.input_parameters["spending_reflection_point"]/self.spending)^(2/(1 - self.input_parameters["scale_up_factor"]))
                 )
            )
        
        return y 
        
        
    def cost_outcome_curve(self): #linear
          
        y = (self.input_paramerers["outcome_full_coverage"] - 
                self.input_parameters["outcome_zero_coverage"]) *\
                self.coverage + self.input_parameters["outcome_zero_coverage"]
                
        return y 
        