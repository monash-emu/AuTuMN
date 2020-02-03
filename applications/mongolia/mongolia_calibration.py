from multiprocess.pool import Pool

from autumn.calibration import *

par_priors = [{'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [10., 20.]},
              {'param_name': 'adult_latency_adjustment', 'distribution': 'uniform', 'distri_params': [2., 6.]},
              {'param_name': 'dr_amplification_prop_among_nonsuccess', 'distribution': 'uniform',
               'distri_params': [.15, .25]},
              {'param_name': 'self_recovery_rate', 'distribution': 'uniform', 'distri_params': [.18, .29]},
              {'param_name': 'tb_mortality_rate', 'distribution': 'uniform', 'distri_params': [.33, .44]},
              {'param_name': 'rr_transmission_recovered', 'distribution': 'uniform', 'distri_params': [.8, 1.2]},
              {'param_name': 'cdr_multiplier', 'distribution': 'uniform', 'distri_params': [.66, 1.5]},
              ]

target_outputs = [{'output_key': 'prevXinfectiousXamong', 'years': [2015.], 'values': [757.], 'cis': [(620., 894.)]},
                  {'output_key': 'prevXlatentXamongXage_5', 'years': [2016.], 'values': [960.], 'cis': [(902., 1018.)]},
                  {'output_key': 'prevXinfectiousXstrain_mdrXamongXinfectious', 'years': [2015.], 'values': [503.],
                   'cis': [(410., 670.)]},
                  {'output_key': 'notifications', 'years': [2015.], 'values': [4685.]}
                  ]

multipliers = {'prevXinfectiousXamong': 1.e5,
               'prevXlatentXamongXage_5': 1.e4,
               'prevXinfectiousXstrain_mdrXamongXinfectious': 1.e4
               }


if __name__ == '__main__':
    n_cpus = 24
    _iterable = []
    for i in range(n_cpus):
        _iterable.append(Calibration(build_mongolia_model, par_priors, target_outputs, multipliers, i))

    def run_a_single_chain(_calib):
        _calib.run_fitting_algorithm(run_mode='autumn_mcmc', n_iterations=100000, n_burned=0,
                                     n_chains=1, available_time=3600.*24.*7.)
        return


    p = Pool(processes=n_cpus)
    p.map(func=run_a_single_chain, iterable=_iterable)
