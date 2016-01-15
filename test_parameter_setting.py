# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 15:51:02 2016

@author: James
"""
import os
import win32com.client as win32
import modules.parameter_estimation as parameter_estimation
reload(parameter_estimation)
import modules.parameter_setting as parameter_setting
reload(parameter_setting)

#parameter_setting.proportion_early_progression.graph_prior()g

## Create evidence document
#vis = True
#os.chdir('..')
#os.chdir('evidence')
#word = win32.Dispatch('Word.Application')
#evidence_document = word.Documents.Add('')
#word.Visible = vis
#word.Selection.TypeText('CURRENT EVIDENCE OBJECTS AVAILABLE FOR USE IN ' +
#                        'MODEL \n\n')
#for pieces_of_evidence in parameter_setting.Evidence:
#    word.Selection.InsertAfter('____________________________________________\n\n')
#    for attribute in pieces_of_evidence.attributes_ordered:
#        word.Selection.InsertAfter(attribute + ': ')
#        word.Selection.InsertAfter(pieces_of_evidence.text[attribute] + '\n\n')
#evidence_document.SaveAs((evidence_directory + 'evidence.docx'))
#evidence_document.Close()
#word = None  # Required to close the document


current_dir = os.path.dirname(__file__)
location = os.path.join(current_dir, '..', 'evidence',
                        self.name + '.pdf')
os.startfile(location)



os.chdir('..')
# Create parameters document
os.chdir('evidence')


word = win32.Dispatch('Word.Application')
parameters_document = word.Documents.Add('')
word.Visible = vis
for parameter_instances in parameter_setting.Parameter:
    parameter_instances.graph_prior()
    word.Selection.TypeText('\n\n')
    for attribute in parameter_instances.attributes_ordered:
        word.Selection.TypeText(attribute + ': ')
        word.Selection.TypeText(parameter_instances.text[attribute] + '\n\n')
    word.Selection.TypeText('\n\n')
    word.Selection.InlineShapes.AddPicture(graph_directory +
                                           parameter_instances.name + '.jpg')
parameters_document.SaveAs((evidence_directory + 'parameters.docx'))
parameters_document.Close()
word = None
os.chdir('..')






