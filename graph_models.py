import os
import autumn.model
import autumn.plotting

os.system('rm -rf models')
os.makedirs('models')

for name, Model in [
        ('single.strain', autumn.model.SingleStrainSystem),
        ('three.strain', autumn.model.ThreeStrainSystem),
        ('three.strain.identi', autumn.model.ThreeStrainFullTreatmentSystem),
    ]:
    print 'running', name
    base = os.path.join('models', name)
    model = Model()
    model.set_flows()
    model.make_graph(base + '.workflow')
    model.make_times(1900, 2050, 0.05)
    model.integrate_explicit()
    autumn.plotting.plot_fractions(model, model.labels, base + '.fraction.png')
    autumn.plotting.plot_populations(model, model.labels, base + '.population.png')

os.system('open models/*png')