# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:14:01 2015

@author: cliffk
"""


def run(command, printinput=False, printoutput=False):
   """ Make it easier to run bash commands """
   from subprocess import Popen, PIPE
   if printinput: print(command)
   try:
       output = Popen(command, shell=True, stdout=PIPE).communicate()[0]
   except:
       output = 'Shell command failed'
   if printoutput: print(output)
   return output


def findinds(val1, val2=None, eps=1e-6):
    """
    Little function to find matches even if two things aren't eactly equal (eg. 
    due to floats vs. ints). If one argument, find nonzero values. With two arguments,
    check for equality using eps. Returns a tuple of arrays if val1 is multidimensional,
    else returns an array.
    
    Examples:
        findinds(rand(10)<0.5) # e.g. array([2, 4, 5, 9])
        findinds([2,3,6,3], 6) # e.g. array([2])
    
    Version: 2014nov27 by cliffk
    """
    from numpy import nonzero, array, ndim
    if val2==None: # Check for equality
        output = nonzero(val1) # If not, just check the truth condition
    else:
        output = nonzero(abs(array(val1)-val2)<eps) # If absolute difference between the two values is less than a certain amount
    if ndim(val1)==1: # Uni-dimensional
        output = output[0] # Return an array rather than a tuple of arrays if one-dimensional
    return output


def sigfig(x, sigfigs=3):
    """ Return a string representation of variable x with sigfigs number of significant figures """
    from numpy import log10, floor
    magnitude = floor(log10(abs(x)))
    factor = 10**(sigfigs-magnitude-1)
    x = round(x*factor)/float(factor)
    digits = int(abs(magnitude) + max(0, sigfigs - max(0,magnitude) - 1) + 1 + (x<0) + (abs(x)<1)) # one because, one for decimal, one for minus
    decimals = int(max(0,-magnitude+sigfigs-1))
    strformat = '%' + '%i.%i' % (digits, decimals)  + 'f'
    string = strformat % x
    return string


def tic():
    """
    A little pair of functions to calculate a time difference, sort of like Matlab:
    t = tic()
    toc(t)
    """
    from time import time
    return time()


def toc(start=0, label='', sigfigs=3):
    """
    A little pair of functions to calculate a time difference, sort of like Matlab:
    t = tic()
    toc(t)
    """
    from time import time
    elapsed = time() - start
    if label=='': base = 'Elapsed time: '
    else: base = 'Elapsed time for %s: ' % label
    print(base + '%s s' % sigfig(elapsed, sigfigs=sigfigs))
    return None
    


def printdata(data, name='Variable', depth=1, maxlen=40, indent='', level=0, showcontents=False):
    """
    Nicely print a complicated data structure, a la Matlab.
    Arguments:
      data: the data to display
      name: the name of the variable (automatically read except for first one)
      depth: how many levels of recursion to follow
      maxlen: number of characters of data to display (if 0, don't show data)
      indent: where to start the indent (used internally)
    
    """
    datatype = type(data)
    def printentry(data):
        from numpy import shape, ndarray
        if datatype==dict: string = ('dict with %i keys' % len(data.keys()))
        elif datatype==list: string = ('list of length %i' % len(data))
        elif datatype==tuple: string = ('tuple of length %i' % len(data))
        elif datatype==ndarray: string = ('array of shape %s' % str(shape(data)))
        elif datatype.__name__=='module': string = ('module with %i components' % len(dir(data)))
        elif datatype.__name__=='class': string = ('class with %i components' % len(dir(data)))
        else: string = datatype.__name__
        if showcontents and maxlen>0:
            datastring = ' | '+str(data)
            if len(datastring)>maxlen: datastring = datastring[:maxlen] + ' <etc> ' + datastring[-maxlen:]
        else: datastring=''
        return string+datastring
    
    string = printentry(data).replace('\n',' \ ') # Remove newlines
    print(level*'..' + indent + name + ' | ' + string)


    if depth>0:
        level += 1
        if type(data)==dict:
            keys = data.keys()
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                thisindent = ' '*(maxkeylen-len(key))
                printdata(data[key], name=key, depth=depth-1, indent=indent+thisindent, level=level)
        elif type(data) in [list, tuple]:
            for i in range(len(data)):
                printdata(data[i], name='[%i]'%i, depth=depth-1, indent=indent, level=level)
        elif type(data).__name__ in ['module', 'class']:
            keys = dir(data)
            maxkeylen = max([len(key) for key in keys])
            for key in keys:
                if key[0]!='_': # Skip these
                    thisindent = ' '*(maxkeylen-len(key))
                    printdata(getattr(data,key), name=key, depth=depth-1, indent=indent+thisindent, level=level)
        print('\n')
    return None



def checkmem(origvariable, descend=0, order='n', plot=False, verbose=0):
    """
    Checks how much memory the variable in question uses by dumping it to file.
    
    Example:
        from utils import checkmem
        checkmem(['spiffy',rand(2483,589)],descend=1)
    """
    from os import getcwd, remove
    from os.path import getsize
    from cPickle import dump
    from numpy import iterable, argsort
    
    filename = getcwd()+'/checkmem.tmp'
    
    def dumpfile(variable):
        wfid = open(filename,'wb')
        dump(variable, wfid)
        return None
    
    printnames = []
    printbytes = []
    printsizes = []
    varnames = []
    variables = []
    if descend==False or not(iterable(origvariable)):
        varnames = ['']
        variables = [origvariable]
    elif descend==1 and iterable(origvariable):
        if hasattr(origvariable,'keys'):
            for key in origvariable.keys():
                varnames.append(key)
                variables.append(origvariable[key])
        else:
            varnames = range(len(origvariable))
            variables = origvariable
    
    for v,variable in enumerate(variables):
        if verbose: print('Processing variable %i of %i' % (v+1, len(variables)))
        dumpfile(variable)
        filesize = getsize(filename)
        factor = 1
        label = 'B'
        labels = ['KB','MB','GB']
        for i,f in enumerate([3,6,9]):
            if filesize>10**f:
                factor = 10**f
                label = labels[i]
        printnames.append(varnames[v])
        printbytes.append(filesize)
        printsizes.append('%0.3f %s' % (float(filesize/float(factor)), label))
        remove(filename)

    if order=='a' or order=='alpha' or order=='alphabetical':
        inds = argsort(printnames)
    else:
        inds = argsort(printbytes)

    for v in inds:
        print('Variable %s is %s' % (printnames[v], printsizes[v]))

    if plot==True:
        from matplotlib.pylab import pie, array, axes
        axes(aspect=1)
        pie(array(printbytes)[inds], labels=array(printnames)[inds], autopct='%0.2f')

    return None