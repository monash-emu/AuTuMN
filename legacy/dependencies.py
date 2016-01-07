# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:47 2015

@author: cliffk
"""

from matplotlib.pylab import zeros, log, imshow, subplot, figure, nan


extension = '.m' # What kind of files to look for cross-dependencies
doprint = False

# Make it easier to run bash commands
def run(command, printinput=False, printoutput=False):
   from subprocess import Popen, PIPE
   if printinput: print(command)
   output = Popen(command, shell=True, stdout=PIPE, executable='/bin/bash').communicate()[0]
   if printoutput: print(output)
   return output

files = run('ls *'+extension).split('\n')[:-1] # Last one is blank
nfiles = len(files)
functions = [i.strip(extension) for i in files]

results = zeros((nfiles,nfiles))
for i,fu in enumerate(functions):
    for j,fi in enumerate(files):
        if i!=j:
            if doprint: print('%s/%s (%i + %i)' % (fu,fi,i,j))
            match = run('cat ' + fi + ' | grep ' + fu)
            if len(match):
                nmatches = len(match.split('\n'))-1
                results[j,i] = log(nmatches)+1
            else:
                results[j,i] = nan
                    

fighandle = figure() # subplot(1,1,1) 
fighandle.subplots_adjust(left=0.3) # Less space on left
fighandle.subplots_adjust(bottom=0.3) # Less space on bottom
plothandle = subplot(1,1,1)
imhandle = imshow(results, interpolation='none')
plothandle.set_xticks(range(nfiles))
plothandle.set_yticks(range(nfiles))
plothandle.set_xticklabels(functions, rotation=90)
plothandle.set_yticklabels(functions)


print('Done.')