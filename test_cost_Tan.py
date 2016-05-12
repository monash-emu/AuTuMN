import os
import glob
import datetime

import numpy
import pylab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


import autumn.base_totestcost
import autumn.model_totestcost
import autumn.plotting_totestcost

import optima.programs 
import optima
import optima.optimization 




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

        allowed_by_types = ["by_fraction", "by_population"]
        if by_type not in allowed_by_types:
            raise Exception(
                "Cant't recognize 'by_type'='%s', should be %s" 
                    % (by_type, allowed_by_types))

        self.costcovfn = optima.programs.Costcov()
        unitcost = calc_unitcost(19E6, 0.5, 160E6)
        #print(unitcost)
        self.costcovfn.addccopar({
            'saturation': (0.75, 0.85),
            't': 2013.0,
            'unitcost': (unitcost-10, unitcost+10)
        })
        self.years = [2013]
        self.param_key = param_key
        self.by_type = by_type
        self.name = name

    def get_coverage(self, budget, popsize):
        coverages = self.costcovfn.evaluate(
            x=numpy.array([budget]), 
            t=numpy.array(self.years), 
            popsize=popsize, 
            bounds=None, 
            toplot=False)
        #print(budget,coverages)
        #pylab.plot(budget,coverages)
        #pylab.show()
        #plt.plot(budget,coverages,"ro")
        #plt.xlabel('Budget')
        #plt.ylabel('Coverage')
        #plt.show()
        return coverages[0]

    def get_param_val(self, budget, popsize):
        val = self.get_coverage(budget, popsize)
        if self.by_type == "by_fraction":
            fraction = val / popsize
            val = fraction
            print(budget,val)
            #plt.plot(budget,val,"ro")
            #plt.show()
        return self.param_key, val



class ModelOptimizer():
    
    def __init__(self, model_totestcost):
        self.model = model_totestcost
        self.programs = []

    def add_program(self, name, param_key, by_type):
        self.programs.append(
            Program(name, param_key, by_type))

    def objective(self, budgetvec):
        popsize = sum(self.model.init_compartments.values())
        for budget, program in zip(budgetvec, self.programs):
            key, val = program.get_param_val(budget, popsize)
            self.model.set_parameter(key, val)
        self.model.integrate_explicit()
        fval = self.model.vars['mortality']
        #print(budget)
        #print " %s -> %s" % (budgetvec, fval)
        return fval


    def optimize_outcome(self):
        constrainedbudgetvec = numpy.array([60E6, 70E6])
        n_program = len(self.programs)
        budgetvecnew, fval, exitflag, output = optima.asd(
              lambda budgetvec: self.objective(budgetvec), 
              constrainedbudgetvec, 
              args={}, 
              xmin=numpy.zeros(n_program), 
              timelimit=None, 
              MaxIter=500, 
              verbose=None)
        #print "Optimized budget %s -> %s" % (budgetvecnew, fval)



model = autumn.base_totestcost.SimpleModel()
model.make_times(1990, 2020, 1)
model_optimizer = ModelOptimizer(model)

"""
model_optimizer.add_program(
    "prog1", "program_rate_detect", "by_fraction")
model_optimizer.add_program(
    "prog2", "program_prop_vac", "by_fraction")
"""

model_optimizer.add_program(
    "pmdt", "program_proportion_success_mdr", "by_fraction")
model_optimizer.add_program(
    "pmdt", "program_proportion_success_xdr", "by_fraction")
model_optimizer.add_program(
    "pmdt", "program_rate_detect_mdr_asmdr", "by_fraction")
model_optimizer.add_program(
    "pmdt", "program_rate_detect_xdr_asxdr", "by_fraction")

model_optimizer.add_program(
    "gene_xpert", "program_rate_detect", "by_fraction")

model_optimizer.add_program(
    "improved_dx", "program_prop_algorithm_sensitivity", "by_fraction")
model_optimizer.add_program(
    "improved_dx", "program_rate_detect", "by_fraction")

model_optimizer.add_program(
    "shortcourse_mdr", "tb_timeperiod_treatment_mdr", "by_fraction")

model_optimizer.add_program(
    "universal_dst", "program_rate_detect_ds_asds", "by_fraction")
model_optimizer.add_program(
    "universal_dst", "program_rate_detect_mdr_asmdr", "by_fraction")
model_optimizer.add_program(
    "universal_dst", "program_rate_detect_xdr_asxdr", "by_fraction")

model_optimizer.optimize_outcome()



#Program 3. Programmatic management of MDR-TB.
#Mechanism: Increase correct identification of MDR-TB cases
        #Increase treatment success
#Affected parameters:
                    # Increase - program_proportion_success_mdr
                    # Increase - program_proportion_success_xdr
                    # Decrease - program_rate_detect_mdr_asds
                    # Increase - program_rate_detect_mdr_asmdr
                    # Decrease - program_rate_detect_mdr_asxdr
                    # Decrease - program_rate_detect_xdr_asds
                    # Decrease - program_rate_detect_xdr_asmdr
                    # Increase - program_rate_detect_xdr_asxdr


#Program 4. Rolling out GeneXpert
#Mechanism: Detect more cases (smear negative TB). The flow will increase from A to D
#thru an increase in smear negative pulmonary TB diagnosis
#Affected parameters:
                    # Increase - program_rate_detect


#Program 5. Improved diagnostic algorithims (this covers: DSSM using microscope, DSSM using FED, line probe assay
#Mechanism: Detect more cases
#Affected parameters:
                    #Increase - program_prop_algorithm_sensitivity
                    #Increase - program_rate_detect


#Program 6. Short course for MDR-TB treatment
#Mechanism: Reduction in duration of treatment, reduced failure and default, increased success rate
#Affected parameters:
                    #Decrease - tb_timeperiod_treatment_mdr


#Program 7. Universal access to rapid drug susceptibility testing
#Mechanism: Increased proportion of people receiving appropriate regimen
         #MDR-TB receives MDR-TB regimen, XDR-TB receives XDR-TB regimen
#Affected parameters:
                    #Increase - program_rate_detect_ds_asds
                    #Decrease - program_rate_detect_mdr_asds
                    #Increase - program_rate_detect_mdr_asmdr
                    #Decrease - program_rate_detect_xdr_asds
                    #Decrease - program_rate_detect_xdr_asmdr
                    #Increase - program_rate_detect_xdr_asxdr



















