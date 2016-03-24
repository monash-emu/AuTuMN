import os
import glob
import datetime

import numpy
from numpy import array, zeros
import autumn.model
import autumn.plotting

import optima.programs 
import optima
from optima.optimization import constrainbudget




def calc_unitcost(population, fraction, budget):
    return budget / (fraction * population)




class Program():
    def __init__(self, name, param_key, by_type):
        """
        Program creates a link between a budget value
        and sets a model parameter.

        Args:
            model: autumn model
            param_key: string of param in model to update by program
            by_type: "by_fraction" or "by_population"
        """

        self.costcovfn = optima.programs.Costcov()
        unitcost = calc_unitcost(19E6, 0.5, 160E6)
        self.costcovfn.addccopar({
            'saturation': (0.75,0.85),
            't': 2013.0,
            'unitcost': (unitcost-10, unitcost+10)
        })
        self.years = [2013]
        self.param_key = param_key
        self.by_type = by_type
        self.name = name

    def get_coverage(self, budget, popsize):
        budgets = [budget]
        coverages = self.costcovfn.evaluate(
            x=array([budget]), 
            t=array(self.years), 
            popsize=popsize, 
            bounds=None, 
            toplot=False)
        return coverages[0]

    def get_param_val(self, budget, popsize):
        val = self.get_coverage(budget, popsize)
        if self.by_type == "by_fraction":
            fraction = val / popsize
            val = fraction
        return self.param_key, val 




class ModelOptimizer():
    
    def __init__(self, model):
        self.model = model
        self.programs = []

    def add_program(self, name, param_key, by_type):
        self.programs.append(
            Program(name, param_key, by_type))

    def objective(self, budgetvec, popsize):

        for budget, program in zip(budgetvec, self.programs):

            key, val = program.get_param_val(budget, popsize)
            print program.name, budget, popsize, key, val
            self.model.set_param(key, val)

        self.model.integrate_explicit()

        return val

    def optimize_outcome(self):

        constrainedbudgetvec = array([60E6, 70E6])

        xmin = zeros(len(constrainedbudgetvec))

        def fn(budgetvec):
            popsize = sum(self.model.init_compartments.values())
            return self.objective(budgetvec, popsize)

        budgetvecnew, fval, exitflag, output = optima.asd(
              fn, 
              constrainedbudgetvec, 
              args={}, 
              xmin=xmin, 
              timelimit=None, 
              MaxIter=500, 
              verbose=None)

        print "Finished"
        print budgetvecnew, fval, exitflag, output
        


model = autumn.model.SimplifiedModel()
model.make_times(1990, 2020, 1)

model_optimizer = ModelOptimizer(model)
model_optimizer.add_program("prog1", "program_rate_detect", "by_fraction")
model_optimizer.add_program("prog2", "program_time_treatment", "by_fraction")
model_optimizer.optimize_outcome()







