# -*- coding: utf-8 -*-
"""
Here I was hoping to write code that would find the modules directory within
AuTuMN from whatever current folder we were in. I think I failed in
this task, although I learnt some coding skills on the way. This code seems
pretty useless now.
Created on Thu Jan 07 19:38:30 2016
@author: James
"""

import os

directory_working = os.getcwd()
subdirectory = 'modules'
# If in a sub-directory of AuTuMN
if directory_working.find('AuTuMN') > 0:
    for a in range(10):
        if directory_working.find('AuTuMN') > 0:
            os.chdir('..')
            directory_working = os.getcwd()
    directory_autumn = directory_working + '\AuTuMN'
    os.chdir(directory_autumn)
    os.chdir(subdirectory)
# If not
else:
    for a in range(10):
        os.chdir('..')
    directory_root = os.getcwd()
    directories = os.listdir(directory_root)
    if 'Users' in directories:
        os.chdir('Users')
        directory_working = os.getcwd()
    else:
        raise Exception('No "Users" directory in the root directory')
    directories = os.listdir(directory_working)
    for a in directories:
        if a[0] == 'J' or a[0] == 'j':
            os.chdir(a)
            directory_working = os.getcwd()
    os.chdir('Desktop\AuTuMN')
    directory_working = os.getcwd()
    os.chdir(subdirectory)
