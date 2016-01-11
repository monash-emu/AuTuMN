# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 15:51:02 2016

@author: James
"""
import os
os.chdir('modules')
import parameter_setting

# A few possible examples follow - uncomment as required

#parameter_setting.early_progression.graph_prior()
#parameter_setting.late_progression.graph_prior()
#parameter_setting.early_latent_duration_sloot2014.open_pdf()
#parameter_setting.untreated_duration_tiemersma2011.open_pdf()
#parameter_setting.early_progression.graph_prior()
#parameter_setting.early_progression_child_trauer2016.write_explanation_document()
#parameter_setting.multiplier_relative_fitness_mdr.graph_prior()

os.chdir('..')
os.chdir('evidence')
file = open('all_evidence.txt', 'w')
file.write('CURRENT EVIDENCE OBJECTS AVAILABLE FOR USE IN MODEL \n\n')
for pieces_of_evidence in parameter_setting.Evidence:
    file.write('_____________________________________________________________')
    file.write(pieces_of_evidence.text)
    print(pieces_of_evidence)
file.close()

file = open('all_parameters.txt', 'w')
file.write('CURRENT BASELINE PARAMETER DISTRIBUTIONS BEING USED IN THE MODEL \n\n')
for parameter_instances in parameter_setting.Parameter:
    file.write('_____________________________________________________________')
    parameter_instances.create_text()
    file.write(parameter_instances.text)
    print(parameter_instances)
file.close()

os.chdir('..')
