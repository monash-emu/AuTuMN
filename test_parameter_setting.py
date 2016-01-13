# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 15:51:02 2016

@author: James
"""
import os
import win32com.client as win32
os.chdir('modules')
import parameter_setting

# A few possible examples follow - uncomment as required
#parameter_setting.proportion_early_progression.graph_prior()
#parameter_setting.proportion_casefatality_active_untreated_smearpos.spread
#parameter_setting.late_progression.graph_prior()
#parameter_setting.early_latent_duration_sloot2014.open_pdf()
#parameter_setting.untreated_duration_tiemersma2011.open_pdf()
#parameter_setting.early_progression.graph_prior()
#parameter_setting.early_progression_child_trauer2016.write_explanation_document()
#parameter_setting.multiplier_relative_fitness_mdr.graph_prior()
#vars(parameter_setting.multiplier_relative_fitness_mdr)
#parameter_setting.proportion_early_progression.limits


os.chdir('..')
os.chdir('evidence')

word = win32.Dispatch('Word.Application')
word.Documents.Open(r'C:/Users/James/Desktop/AuTuMN/evidence/evidence.docx')
word.Selection.TypeText('CURRENT EVIDENCE OBJECTS AVAILABLE FOR USE IN MODEL \n\n')
for pieces_of_evidence in parameter_setting.Evidence:
    word.Selection.TypeText('________________________________________________')
    word.Selection.TypeText(pieces_of_evidence.text)
word = None
os.chdir('..')

word = win32.Dispatch('Word.Application')
word.Documents.Open(r'C:/Users/James/Desktop/AuTuMN/evidence/parameters.docx')
word.Selection.TypeText('CURRENT BASELINE PARAMETER DISTRIBUTIONS BEING ' +
    'USED IN THE MODEL \n\n')
for parameter_instances in parameter_setting.Parameter:
    parameter_instances.create_text()
    word.Selection.TypeText('________________________________________________')
    word.Selection.TypeText(parameter_instances.text)
word = None
os.chdir('..')

