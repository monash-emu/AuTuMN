# -*- coding: utf-8 -*-

"""
Created on Sat Nov 28 12:01:25 2015

@author: James
"""

from parameter_estimation import Parameter, Evidence

# EVIDENCE_____________________________________________________________________
early_progression_trauer2016 \
    = Evidence('Early progression proportion', 0.125, [.096, 0.154],
               'Early progression proportion, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'Results describe 12.5% one year risk accruing over ' +
               'first 227 days. Confidence interval currently estimated. ',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_trauer2016 \
    = Evidence('Early progression proportion - children', 0.42, [0.308, 0.532],
               'Early progression proportion in children, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'Estimated pooled results for both paediatric groups.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_trauer2016 \
    = Evidence('Early progression proportion - adults', 0.024, [0.009, 0.0393],
               'Early progression proportion in adults, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'See comments for all-age evidence object.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

early_progression_sloot2014 \
    = Evidence('Early progression proportion', 0.095, [0.074, 0.116],
               'Early progression proportion, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, approximately 9 to 10% of the 739 contacts ' +
               'with evidence of infection developed active TB in the early ' +
               'high risk period, which lasted for around two years. ' +
               'Confidence interval estimated from graphs.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_sloot2014 \
    = Evidence('Early progression proportion - children', 0.26, [0.203, 0.317],
               'Early progression proportion in children, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'Estimated from Figure 3.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_sloot2014 \
    = Evidence('Early progression proportion - adults', 0.07, [0.055, 0.085],
               'Early progression proportion in adults, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'Estimated from Figure 3.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

early_progression_diel2011 \
    = Evidence('Early progression proportion', 19. / 147., [],
               'Early progression proportion, Diel et al. 2011',
               'early_progression_diel2011.pdf',
               'Table 4: 19 of 147 untreated QFT-positive individuals got TB' +
               'over two years of follow up',
               'Diel R, Loddenkemper R, Niemann S, Meywald-Walter K, ' +
               'Nienhaus A. Negative and Positive Predictive Value of a ' +
               'Whole-Blood Inteferon-gamma Release Assay for Developing ' +
               'Active Tuberculosis. Am J Respir Crit Care Med. ' +
               '2010;183(1):88-95.')

early_latent_duration_trauer2016 \
    = Evidence('Early latent duration', 5. / 12., [],
               'Early latent duration, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')
early_latent_duration_child_trauer2016 \
    = Evidence('Early latent duration in children', 145. / 365., [],
               'Early latent duration in children, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')
early_latent_duration_adult_trauer2016 \
    = Evidence('Early latent duration in adults', 227. / 365., [],
               'Early latent duration in adults, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')

early_latent_duration_sloot2014 \
    = Evidence('Early latent duration', 1.8, [],
               'Early latent duration, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_child_sloot2014 \
    = Evidence('Early latent duration in children', 0.6, [],
               'Early latent duration in children, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_adult_sloot2014 \
    = Evidence('Early latent duration in adults', 1.8, [],
               'Early latent duration in adults, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

late_progression_rate_horsburgh2010 \
    = Evidence('Late progression rate', 0.00058, [0.00038, 0.00089],
               'Late progression rate, Horsburgh et al. 2010',
               'late_progression_horsburgh2010.pdf',
               'Main finding is rate of reactivation of 0.04 per year',
               'Community based estimate of rate of reactivation from ' +
               'the US.')

bcg_protection_colditz \
    = Evidence('BCG protection', 0.49, [0.34, 0.70],
               'Relative risk of TB, Colditz et al. 1994',
               'bcg_protection_colditz1994.pdf',
               'Estimate of relative risk of TB in BCG vaccinated. ',
               'Confidence interval provided for RR estimate.')

untreated_duration_tiemersma2011 \
    = Evidence('Untreated duration', 3, [],
               'Untreated duration, Tiemersma et al. 2011',
               'natural_history_tiemersma2011.pdf',
               'Estimate from pre-chemotherapy literature of three years ' +
               'untreated', 'No confidence interval around this estimate')

untreated_casefatality_smearpos_tiemersma2011 \
    = Evidence('Untreated case fatality', .7, [],
               'Untreated case fatality, Tiemersma et al. 2011',
               'natural_history_tiemersma2011.pdf',
               '70% case fatality for smear-positive patients',
               'Review of the pre-chemotherapy literature.')

treatment_duration_ds_who2011 \
    = Evidence('Time under treatment for DS-TB', 6., [.5],
               'Time to complete treatment for DS-TB based on WHO guidelines',
               'treatmentguidelines_who2011.pdf',
               'World Health Organization. Guidelines on the programmatic ' +
               'management of drug-resistant tuberculosis. Available at: ' +
               'http://apps.who.int/iris/bitstream/10665/44597/1/' +
               '97892415101583_eng.pdf',
               'Based on recommended treatment durations')
treatment_duration_mdr_who2011 \
    = Evidence('Time under treatment for DS-TB', 20., [6.],
               'Time to complete treatment for DS-TB based on WHO guidelines',
               'treatmentguidelines_who2011.pdf',
               'World Health Organization. Guidelines on the programmatic ' +
               'management of drug-resistant tuberculosis. Available at: ' +
               'http://apps.who.int/iris/bitstream/10665/44597/1/' +
               '97892415101583_eng.pdf',
               'Based on recommended treatment durations')

# PARAMETERS___________________________________________________________________
proportion_early_progression \
    = Parameter('Early progression proportion',
                'proportion',
                'beta_full_range',
                early_progression_trauer2016.estimate,
                early_progression_trauer2016.interval, [],
                ['latent_early to active',
                 'latent_early to latent_late'])
proportion_early_progression_child \
    = Parameter('Early progression proportion in children',
                'proportion',
                'beta_full_range',
                early_progression_child_trauer2016.estimate,
                early_progression_child_trauer2016.interval, [],
                ['latent_early to active (children)',
                 'latent_early to latent_late (children)'])
proportion_early_progression_adult \
    = Parameter('Early progression proportion in adults',
                'proportion',
                'beta_full_range',
                early_progression_adult_trauer2016.estimate,
                early_progression_adult_trauer2016.interval, [],
                ['latent_early to active (adults)',
                 'latent_early to latent_late (adults)'])

timeperiod_early_latent \
    = Parameter('Time spent in early latency',
                'timeperiod',
                'gamma',
                early_latent_duration_trauer2016.estimate,
                early_latent_duration_trauer2016.interval, [],
                ['early_latent (if not reinfected)'])
timeperiod_early_latent_child \
    = Parameter('Time spent in early latency - children',
                'timeperiod',
                'gamma',
                early_latent_duration_child_trauer2016.estimate,
                early_latent_duration_child_trauer2016.interval, [],
                ['early_latent (children, if not reinfected)'])
timeperiod_early_latent_adult \
    = Parameter('Time spent in early latency - adults',
                'timeperiod',
                'gamma',
                early_latent_duration_adult_trauer2016.estimate,
                early_latent_duration_adult_trauer2016.interval, [],
                ['early_latent (adults, if not reinfected)'])

rate_late_progression \
    = Parameter('Late progression rate',
                'rate',
                'gamma',
                late_progression_rate_horsburgh2010.estimate,
                late_progression_rate_horsburgh2010.interval, [],
                ['late_latent', 'active'])

timeperiod_activeuntreated \
    = Parameter('Untreated duration',
                'timeperiod',
                'gamma',
                untreated_duration_tiemersma2011.estimate,
                untreated_duration_tiemersma2011.interval, [],
                ['active disease (if not detected)'])

proportion_casefatality_active_untreated_smearpos \
    = Parameter('Untreated case fatality',
                'proportion',
                'beta_full_range',
                untreated_casefatality_smearpos_tiemersma2011.estimate,
                untreated_casefatality_smearpos_tiemersma2011.interval, [],
                ['death', 'death and spontaneous recovery (if not detected)'])

multiplier_relative_fitness_mdr \
    = Parameter('Relative fitness',
                'multiplier',
                'beta_full_range',
                0.6, [0.25], [],
                ['force of infection'])

mutliplier_bcg_protection \
    = Parameter('BCG protection',
                'multiplier',
                'beta_full_range',
                bcg_protection_colditz.estimate,
                bcg_protection_colditz.interval, [],
                ['force of infection parameter'])

timeperiod_treatment_ds \
    = Parameter('Time under treatment for DS-TB',
                'timeperiod',
                'normal_truncated',
                treatment_duration_ds_who2011.estimate,
                treatment_duration_ds_who2011.interval,
                [treatment_duration_ds_who2011.estimate, 1e3],
                ['time under treatment for DS-TB (may span multiple compartments)'])
timeperiod_treatment_mdr \
    = Parameter('Time under treatment for MDR-TB',
                'timeperiod',
                'normal_truncated',
                treatment_duration_mdr_who2011.estimate,
                treatment_duration_mdr_who2011.interval,
                [treatment_duration_mdr_who2011.estimate, 1e3],
                ['time under treatment for MDR-TB (may span multiple compartments)'])




