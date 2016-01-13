# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 22:23:28 2016

@author: James
"""

import win32com.client
word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.Documents.Open(r'C:/Users/James/Desktop/AuTuMN/evidence/evidence.docx')

#%%
word.Selection.TypeText('something')

#%%

word.Quit()
word = None


#%%



word = win32com.client.Dispatch('Word.Application')
document = word.Documents.Open(r'C:/Users/James/Desktop/AuTuMN/evidence/evidence.docx')
word.Selection.TypeText('something')

word = None