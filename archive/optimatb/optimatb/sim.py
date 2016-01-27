from matplotlib.pylab import zeros, ones
from dict_equal import dict_equal

class Sim(object):
    def __init__(self,meta,pars,name='Default'):
        # NB argument list will change once Project/Region object is created
        self.name = name
        self.meta = meta # Will eventually be retrieved from the region
        self.pars = pars
        self.results = None
        self.parsmodel = {}   
        self.initialized = False

    def __eq__(self,other):
        a = self.__dict__
        b = other.__dict__

        # Skip certain fields - need a better way to do this
        a.pop('name')
        b.pop('name')
        a['meta'].pop("version", None)
        a['meta'].pop("commit", None)
        a['meta'].pop("date", None)
        b['meta'].pop("version", None)
        b['meta'].pop("commit", None)
        b['meta'].pop("date", None)

        return dict_equal(a,b)

    def run(self):
        from model import model
        if ~self.initialized:
            self.initialize()
        self.results = model(self)

    def initialize(self):
        # This function prepares the sim so that it is in a runnable state
        # i.e. self.run() should work properly if executed after self.initialize()
        self.makemodelpars()
        self.initialized = True

    def makemodelpars(self):
        # Set self.parsmodel to the format that model.py requires
        keys = ['birth','death','tbdeath','treatdeath','ncontacts','progress','treat','recov','test' ] 
        self.parsmodel = {}  # Clear whatever is already in here 
        for k in keys:
            self.parsmodel[k] = self.pars[k]*ones(self.meta['npts'])

    def initialconditions(self):
        """ Initialize the output structure """
        people = zeros((self.meta['nstates'], self.meta['npts']))
        people[:,0] = self.meta['initial'] # Initial conditions
        return people

    def plot(self,show_wait = True):
        from matplotlib.pylab import figure, plot, legend, xlabel, ylabel, subplot, show
        from gridcolormap import gridcolormap
        figh = figure()
        figh.subplots_adjust(right=0.7)
        
        subplot(2,1,1)
        colors = gridcolormap(self.meta['nstates'])
        for s in xrange(self.meta['nstates']):
            plot(self.meta['tvec'], self.results['people'][s,:]/self.results['people'][:,:].sum(axis=0), linewidth=2, color=colors[s])
        
        xlabel('Time (years)')
        ylabel('Fraction of people')
        legend(self.meta['statenames'], bbox_to_anchor=(1.5, 1.00))
        
        subplot(2,1,2)
        legendkeys = []
        colors = gridcolormap(len(self.results.keys()))
        for i,k in enumerate(self.results.keys()):
            if k!='people':
                plot(self.meta['tvec'], self.results[k], linewidth=2, color=colors[i])
                legendkeys.append(k)
        
        xlabel('Time (years)')
        ylabel('Number of people')
        legend(legendkeys, bbox_to_anchor=(1.5, 1.00))

        if show_wait:
            show()

        return None
