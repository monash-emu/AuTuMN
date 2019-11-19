from autumn_from_summer.calibration import *

par_priors = [{'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [1., 20.]},
                  # {'param_name': 'rr_transmission_ger', 'distribution': 'uniform', 'distri_params': [1., 5.]},
                  # {'param_name': 'rr_transmission_urban', 'distribution': 'uniform', 'distri_params': [1., 5.]},
                  # {'param_name': 'rr_transmission_province', 'distribution': 'uniform', 'distri_params': [.5, 5.]},
                  # {'param_name': 'latency_adjustment', 'distribution': 'uniform', 'distri_params': [1., 3.]},
                  # {'param_name': 'dr_amplification_prop_among_nonsuccess', 'distribution': 'uniform',
                  #  'distri_params': [.05, .20]}
                  ]

target_outputs = [{'output_key': 'prevXinfectiousXamongXage_15Xage_60', 'years': [2015.], 'values': [560.]},
                  # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xhousing_ger', 'years': [2015.], 'values': [613.]},
                  # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xlocation_urban', 'years': [2015.], 'values': [586.]},
                  # {'output_key': 'prevXinfectiousXamongXage_15Xage_60Xlocation_province', 'years': [2015.], 'values': [513.]},
                  # {'output_key': 'prevXlatentXamongXage_5', 'years': [2016.], 'values': [960.]},
                  # {'output_key': 'prevXinfectiousXstrain_mdrXamongXinfectious', 'years': [2015.], 'values': [500]}
                 ]

multipliers = {'prevXlatentXamongXage_5': 1.e4,
               'prevXinfectiousXstrain_mdrXamongXinfectious': 1.e4
               }

for i, output in enumerate(target_outputs):
    if output['output_key'][0:15] == 'prevXinfectious' and \
            output['output_key'] != 'prevXinfectiousXstrain_mdrXamongXinfectious':
        multipliers[output['output_key']] = 1.e5

calib = Calibration(build_mongolia_model, par_priors, target_outputs, multipliers)
calib.run_fitting_algorithm(run_mode='mcmc', mcmc_method='Metropolis', n_iterations=2, n_burned=0,
                            n_chains=1, parallel=False)

print(calib.mcmc_trace)
