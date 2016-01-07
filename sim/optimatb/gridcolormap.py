"""
GRIDCOLORMAP

Create a qualitative colormap by assigning points according to the minimum pairwise distance.

Usage example:
from matplotlib.pylab import *
from gridcolormap import gridcolormap
ncolors = 10
piedata = rand(ncolors)
colors = gridcolormap(ncolors, doplot=True)
figure()
pie(piedata, colors=colors)
show()

Version: 2015may13 by cliffk
"""

def gridcolormap(npts=10, limits=None, nsteps=10, doplot=False, asarray=False):

    ## Imports
    from numpy import linspace, meshgrid, array, transpose, inf, zeros, argmax, minimum
    from numpy.linalg import norm
    
    ## Calculate sliding limits if none provided
    if limits is None:
        colorrange = 1-1/float(npts**0.5)
        limits = [0.5-colorrange/2, 0.5+colorrange/2]
    
    ## Calculate primitives and dot locations
    primitive = linspace(limits[0], limits[1], nsteps) # Define primitive color vector
    x, y, z = meshgrid(primitive, primitive, primitive) # Create grid of all possible points
    dots = transpose(array([x.flatten(), y.flatten(), z.flatten()])) # Flatten into an array of dots
    ndots = nsteps**3 # Calculate the number of dots
    indices = [0] # Initialize the array
    
    ## Calculate the distances
    for pt in range(npts-1): # Loop over each point
        totaldistances = inf+zeros(ndots) # Initialize distances
        for ind in indices: # Loop over each existing point
            rgbdistances = dots - dots[ind] # Calculate the distance in RGB space
            totaldistances = minimum(totaldistances, norm(rgbdistances,axis=1)) # Calculate the minimum Euclidean distance
        maxindex = argmax(totaldistances) # Find the point that maximizes the minimum distance
        indices.append(maxindex) # Append this index
    
    ## Wrap up: optionally turn into a list of tuples
    if asarray:
        colors = []
        for i in indices: colors.append(tuple(dots[i,:])) # Gather output
    else:
        colors = dots[indices,:]
    
    ## For plotting
    if doplot:
        from mpl_toolkits.mplot3d import Axes3D # analysis:ignore
        from matplotlib.pyplot import figure
        fig = figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.cla()
        ax.scatter(dots[indices,0], dots[indices,1], dots[indices,2], c=colors, s=200, depthshade=False)
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_zlim((0,1))
        ax.grid(False)
    
    return colors
    
    
