# This function demonstrates how to write a standalone plotting function
# within the OOP framework - the idea is that these types of functions
# take in objects such as Sims, rather than parts of the objects e.g. Sim.results
def plot_overlay(sims,show_wait = True):
    from matplotlib.pylab import figure, plot, legend, xlabel, ylabel, subplot, show
    from gridcolormap import gridcolormap
    
    figh = figure()
    figh.subplots_adjust(right=0.7)
    
    subplots = sims[0].results.keys()
    subplots.remove('people')
    sim_names = [x.name for x in sims]

    for i in xrange(len(subplots)):
        subplot(len(subplots),1,i+1)
        colors = gridcolormap(len(sims))

        for j,sim in enumerate(sims):
            plot(sim.meta['tvec'], sim.results[subplots[i]], linewidth=2, color=colors[j])
            xlabel('Time (years)')
            ylabel(subplots[i])
        
        legend(sim_names, bbox_to_anchor=(1.5, 1.00))

    if show_wait:
        show()
