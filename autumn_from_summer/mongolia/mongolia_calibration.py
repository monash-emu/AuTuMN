from autumn_from_summer.calibration import *

par_priors = [{'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [1., 20.]},
              {'param_name': 'prop_smearpos', 'distribution': 'uniform', 'distri_params': [.25, .5]},
              {'param_name': 'rr_transmission_urban_nonger', 'distribution': 'uniform', 'distri_params': [.1, 10.]},
              {'param_name': 'rr_transmission_urban_ger', 'distribution': 'uniform', 'distri_params': [.1, 10.]},
              # {'param_name': 'rr_transmission_mine', 'distribution': 'uniform', 'distri_params': [1, 10.]},
              {'param_name': 'rr_transmission_prison', 'distribution': 'uniform', 'distri_params': [1., 10.]},
              {'param_name': 'latency_adjustment', 'distribution': 'uniform', 'distri_params': [.3, 3.]},
              {'param_name': 'dr_amplification_prop_among_nonsuccess', 'distribution': 'uniform',
               'distri_params': [.05, .20]}
              ]

target_outputs = [{'output_key': 'prevXinfectiousXorgan_smearposXamongXage_15Xage_60', 'years': [2015.], 'values': [204.],
                   'cis': [(143., 265.1)]},
                  {'output_key': 'prevXinfectiousXorgan_smearnegXamongXage_15Xage_60', 'years': [2015.], 'values': [340.],
                   'cis': [(273., 407.)]},
                  # {'output_key': 'prevXinfectiousXorgan_smearposXlocation_rural_provinceXamongXage_15Xage_60', 'years': [2015.], 'values': [220.]},
                  {'output_key': 'prevXinfectiousXorgan_smearposXlocation_urban_gerXamongXage_15Xage_60',
                   'years': [2015.], 'values': [277.]},
                  {'output_key': 'prevXinfectiousXorgan_smearposXlocation_urban_nongerXamongXage_15Xage_60', 'years': [2015.], 'values': [156]},
                  {'output_key': 'prevXinfectiousXlocation_prisonXamongXage_15Xage_60', 'years': [2015.], 'values': [3785]},
                  {'output_key': 'prevXlatentXamongXage_5', 'years': [2016.], 'values': [960.], 'cis': [(902., 1018.)]},
                  {'output_key': 'prevXinfectiousXstrain_mdrXamongXinfectious', 'years': [2015.], 'values': [500]}
                  ]

multipliers = {'prevXlatentXamongXage_5': 1.e4,
               'prevXinfectiousXstrain_mdrXamongXinfectious': 1.e4
               }

load = False

if not load:
    for i, output in enumerate(target_outputs):
        if output['output_key'][0:15] == 'prevXinfectious' and \
                output['output_key'] != 'prevXinfectiousXstrain_mdrXamongXinfectious':
            multipliers[output['output_key']] = 1.e5

    calib = Calibration(build_mongolia_model, par_priors, target_outputs, multipliers)
    calib.run_fitting_algorithm(run_mode='autumn_mcmc', n_iterations=4, n_burned=0,
                                n_chains=1, available_time=3600.*12)

    print(calib.mcmc_trace)
else:
    models = load_calibration_from_db('outputs_11_27_2019_14_07_54.db')
