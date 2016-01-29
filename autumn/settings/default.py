# -*- coding: utf-8 -*-

"""
Created on Sat Nov 28 12:01:25 2015

@author: James
"""

from ..parameter import Parameter, Evidence


# EVIDENCE_____________________________________________________________________
early_progression_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early progression proportion', 0.125, [.096, 0.154],
               'Early progression proportion, Trauer et al. 2016',
               'Results describe 12.5% one year risk accruing over ' +
               'first 227 days. Confidence interval currently estimated. ',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early progression proportion - children', 0.42, [0.308, 0.532],
               'Early progression proportion in children, Trauer et al. 2016',
               'Estimated pooled results for both paediatric groups.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early progression proportion - adults', 0.024, [0.009, 0.0393],
               'Early progression proportion in adults, Trauer et al. 2016',
               'See comments for all-age evidence object.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

early_progression_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early progression proportion', 0.095, [0.074, 0.116],
               'Early progression proportion, Sloot et al. 2014',
               'From Figure 2, approximately 9 to 10% of the 739 contacts ' +
               'with evidence of infection developed active TB in the early ' +
               'high risk period, which lasted for around two years. ' +
               'Confidence interval estimated from graphs.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early progression proportion - children', 0.26, [0.203, 0.317],
               'Early progression proportion in children, Sloot et al. 2014',
               'Estimated from Figure 3.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early progression proportion - adults', 0.07, [0.055, 0.085],
               'Early progression proportion in adults, Sloot et al. 2014',
               'Estimated from Figure 3.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

early_progression_diel2011 \
    = Evidence('early_progression_diel2011',
               'Early progression proportion', 19. / 147., [],
               'Early progression proportion, Diel et al. 2011',
               'Table 4: 19 of 147 untreated QFT-positive individuals got TB' +
               'over two years of follow up',
               'Diel R, Loddenkemper R, Niemann S, Meywald-Walter K, ' +
               'Nienhaus A. Negative and Positive Predictive Value of a ' +
               'Whole-Blood Inteferon-gamma Release Assay for Developing ' +
               'Active Tuberculosis. Am J Respir Crit Care Med. ' +
               '2010;183(1):88-95.')

early_latent_duration_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early latent duration', 5. / 12., [],
               'Early latent duration, Trauer et al. 2016',
               'From the text of our Results (first paragraph).',
               'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
               'McBryde ES, Denholm JT. ' +
               'Risk of Active Tuberculosis in the Five Years Following ' +
               'Infection . . . 15%' +
               'Chest. 2016;149(2):516-525.')
early_latent_duration_child_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early latent duration in children', 145. / 365., [],
               'Early latent duration in children, Trauer et al. 2016',
               'From the text of our Results (first paragraph).',
               'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
               'McBryde ES, Denholm JT. ' +
               'Risk of Active Tuberculosis in the Five Years Following ' +
               'Infection . . . 15%' +
               'Chest. 2016;149(2):516-525.')
early_latent_duration_adult_trauer2016 \
    = Evidence('early_progression_trauer2016',
               'Early latent duration in adults', 227. / 365., [],
               'Early latent duration in adults, Trauer et al. 2016',
               'From the text of our Results (first paragraph).',
               'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
               'McBryde ES, Denholm JT. ' +
               'Risk of Active Tuberculosis in the Five Years Following ' +
               'Infection . . . 15%' +
               'Chest. 2016;149(2):516-525.')

early_latent_duration_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early latent duration', 1.8, [],
               'Early latent duration, Sloot et al. 2014',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_child_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early latent duration in children', 0.6, [],
               'Early latent duration in children, Sloot et al. 2014',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_adult_sloot2014 \
    = Evidence('early_progression_sloot2014',
               'Early latent duration in adults', 1.8, [],
               'Early latent duration in adults, Sloot et al. 2014',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

late_progression_rate_horsburgh2010 \
    = Evidence('late_progression_horsburgh2010',
               'Late progression rate', 0.00058, [0.00038, 0.00089],
               'Late progression rate, Horsburgh et al. 2010',
               'Main finding is rate of reactivation of 0.04 per year from ' +
               'a community based estimate of rate of reactivation from ' +
               'the US.',
               'Horsburgh CR, O-Donnell M, Chamblee S, et al. ' +
               'Revisiting Rates of Reactivation Tuberculosis. ' +
               'Am J Respir Crit Care Med 182:420-425.')

bcg_protection_colditz \
    = Evidence('bcg_protection_colditz1994',
               'BCG protection', 0.49, [0.34, 0.70],
               'Relative risk of TB, Colditz et al. 1994',
               'Estimate of relative risk of TB in BCG vaccinated. ',
               'Confidence interval provided for RR estimate.')

untreated_duration_tiemersma2011 \
    = Evidence('natural_history_tiemersma2011',
               'Untreated duration', 3, [],
               'Untreated duration, Tiemersma et al. 2011',
               'Estimate from pre-chemotherapy literature of three years ' +
               'untreated', 'No confidence interval around this estimate')

untreated_casefatality_smearpos_tiemersma2011 \
    = Evidence('natural_history_tiemersma2011',
               'Untreated case fatality', .7, [],
               'Untreated case fatality, Tiemersma et al. 2011',
               '70% case fatality for smear-positive patients from ' +
               'Review of the pre-chemotherapy literature.',
               'Tiemersma EW, van der Werf MJ, Borgdorff MW, et al. ' +
               'Natural history of tuberculosis: duration and fatality of ' +
               'untreated pulmonary tuberculosis in HIV negative patients: ' +
               'a systematic review. PLoS One 6(4):e17601.')

untreated_casefatality_smearneg_tiemersma2011 \
    = Evidence('natural_history_tiemersma2011',
               'Untreated case fatality', .2, [],
               'Untreated case fatality, Tiemersma et al. 2011',
               '20% case fatality for smear-negative patients from ' +
               'Review of the pre-chemotherapy literature.',
               'Tiemersma EW, van der Werf MJ, Borgdorff MW, et al. ' +
               'Natural history of tuberculosis: duration and fatality of ' +
               'untreated pulmonary tuberculosis in HIV negative patients: ' +
               'a systematic review. PLoS One 6(4):e17601.')

treatment_duration_ds_who2011 \
    = Evidence('treatmentguidelines_who2011',
               'Time under treatment for DS-TB', 6. / 12., [.5 / 12.],
               'Time to complete treatment for DS-TB based on WHO guidelines',
               'Based on recommended treatment durations',
               'World Health Organization. Guidelines on the programmatic ' +
               'management of drug-resistant tuberculosis. Available at: ' +
               'http://apps.who.int/iris/bitstream/10665/44597/1/' +
               '97892415101583_eng.pdf')
treatment_duration_mdr_who2011 \
    = Evidence('treatmentguidelines_who2011',
               'Time under treatment for DS-TB', 20. / 12., [6. / 12.],
               'Time to complete treatment for DS-TB based on WHO guidelines',
               'Based on recommended treatment durations',
               'World Health Organization. Guidelines on the programmatic ' +
               'management of drug-resistant tuberculosis. Available at: ' +
               'http://apps.who.int/iris/bitstream/10665/44597/1/' +
               '97892415101583_eng.pdf')

smearneg_transmission_tostmann2008 \
    = Evidence('smearneg_transmission_tostmann2008',
               'Relative transmissibility of smear negative disease',
               0.24, [0.2, 0.3],
               'Chance of transmission for smear negatives compared to ' +
               'smear positives, Tostmann et al. 2008',
               'Well constructed study looking at exactly this parameter ' +
               'and employing genotypic confirmation of transmission. ' +
               '(Possible ref for extrapulmonary not being transmissible.)',
               'Tostmann A, Kik SV, Kaisvaart NA, et al. ' +
               'Tuberculosis Transmission by Patients with Smear-Negative ' +
               'Pulmonary Tuberculosis in a Large Cohort in the ' +
               'Netherlands. Clin Infect Dis 2008;47:1135-1142.')

undertreatment_transmission_ds_dhamardhikari2014 \
    = Evidence('undertreatment_transmission_ds_dhamardhikari2014',
               'Relative transmissibility of treatment patients (DS-TB)',
               2. / 100., [],
               'Chance of transmission of treated patients compared to ' +
               'untreated, Dhamardhikari et al. 2014',
               '50 times greater rate of guinea pig TST conversion for ' +
               'untreated patients than treated (DS-TB)',
               'Dhamardhikari AS, Mphahlele M, Venter K, et al. ' +
               'Rapid impact of effective treatment on transmission of ' +
               'multidrug-resistant tuberculosis. Int J Tuberc Lung Dis ' +
               '2014;18(9):1019-1025.')

undertreatment_transmission_mdr_dhamardhikari2014 \
    = Evidence('undertreatment_transmission_ds_dhamardhikari2014',
               'Relative transmissibility of treatment patients (MDR-TB)',
               2. / 28., [],
               'Chance of transmission of treated patients compared to ' +
               'untreated, Dhamardhikari et al. 2014',
               'Sixfold greater rate of guinea pig TST conversion for ' +
               'untreated patients than treated (MDR-TB)',
               'Dhamardhikari AS, Mphahlele M, Venter K, et al. ' +
               'Rapid impact of effective treatment on transmission of ' +
               'multidrug-resistant tuberculosis. Int J Tuberc Lung Dis ' +
               '2014;18(9):1019-1025.')

amplificiation_tomdr_bonnet2011 \
    = Evidence('amplification_tomdr_bonnet2011',
               'Proportion of defaults resulting in amplification to MDR',
               2. / 15., [],
               'Proportion of defaults amplifying, Bonnet et al. 2011',
               'Among 15 non-MDR-TB patients with negative treatment ' +
               'outcomes, two amplified to MDR-TB.',
               'Bonnet M, Pardini M, Meacci F, et al. ' +
               'Treatment of Tuberculosis in a Region with High Drug ' +
               'Resistance: Outcomes, Drug Resistance Amplification and ' +
               'Re-Infection. PLoS One 2011;6(8):e23081.')

negativeoutcome_retreatment_ds_espinal2000 \
    = Evidence('negativeoutcome_retreatment_ds_espinal2000',
               'Greater rate of negative outcomes in retreatment DS cases',
               33. / 15., [],
               'Proportionately greater rate of negative outcomes in ' +
               'retreatment cases compared to new cases, for appropriately ' +
               'treated DS-TB patients, Espinal et al. 2000',
               '85% treatment success in pan-susceptible new TB patients, ' +
               'compared to 67% success in pan-susceptible treatment patients',
               'Espinal MA, Kim SJ, Suarez PG, et al. Standard Short-Course ' +
               'Chemotherapy for Drug-Resistant Tuberculosis. ' +
               'JAMA 2000;283(19):2537-45')

inappropriatetreatment_new_mdr_espinal2000 \
    = Evidence('negativeoutcome_retreatment_ds_espinal2000',
               'Treatment success inappropriately treated new MDR',
               0.52, [],
               'Treatment success rate for inappropriately treated new ' +
               'MDR cases, Espinal et al. 2000',
               '52% treatment success in new MDR-TB cases treated with ' +
               'a standard first line regimen',
               'Espinal MA, Kim SJ, Suarez PG, et al. Standard Short-Course ' +
               'Chemotherapy for Drug-Resistant Tuberculosis. ' +
               'JAMA 2000;283(19):2537-45')

inappropriatetreatment_retreatment_mdr_espinal2000 \
    = Evidence('negativeoutcome_retreatment_ds_espinal2000',
               'Treatment success inappropriately treated retreatment MDR',
               0.29, [],
               'Treatment success rate for inappropriately treated '
               'retreatment MDR cases, Espinal et al. 2000',
               '29% treatment success in new MDR-TB cases treated with ' +
               'a standard first line regimen',
               'Espinal MA, Kim SJ, Suarez PG, et al. Standard Short-Course ' +
               'Chemotherapy for Drug-Resistant Tuberculosis. ' +
               'JAMA 2000;283(19):2537-45')

# PARAMETERS___________________________________________________________________
proportion_early_progression \
    = Parameter('proportion_early_progression',
                'Early progression proportion',
                'proportion',
                'beta_full_range',
                early_progression_trauer2016.estimate,
                early_progression_trauer2016.interval, [],
                ['latent_early to active',
                 'latent_early to latent_late'])
proportion_early_progression_child \
    = Parameter('proportion_early_progression_child',
                'Early progression proportion in children',
                'proportion',
                'beta_full_range',
                early_progression_child_trauer2016.estimate,
                early_progression_child_trauer2016.interval, [],
                ['latent_early to active (children)',
                 'latent_early to latent_late (children)'])
proportion_early_progression_adult \
    = Parameter('proportion_early_progression_adult',
                'Early progression proportion in adults',
                'proportion',
                'beta_full_range',
                early_progression_adult_trauer2016.estimate,
                early_progression_adult_trauer2016.interval, [],
                ['latent_early to active (adults)',
                 'latent_early to latent_late (adults)'])

timeperiod_early_latent \
    = Parameter('timeperiod_early_latent',
                'Time spent in early latency',
                'timeperiod',
                'gamma',
                early_latent_duration_trauer2016.estimate,
                early_latent_duration_trauer2016.interval, [],
                ['early_latent (if not reinfected)'])
timeperiod_early_latent_child \
    = Parameter('timeperiod_early_latent_child',
                'Time spent in early latency - children',
                'timeperiod',
                'gamma',
                early_latent_duration_child_trauer2016.estimate,
                early_latent_duration_child_trauer2016.interval, [],
                ['early_latent (children, if not reinfected)'])
timeperiod_early_latent_adult \
    = Parameter('timeperiod_early_latent_adult',
                'Time spent in early latency - adults',
                'timeperiod',
                'gamma',
                early_latent_duration_adult_trauer2016.estimate,
                early_latent_duration_adult_trauer2016.interval, [],
                ['early_latent (adults, if not reinfected)'])

rate_late_progression \
    = Parameter('rate_late_progression',
                'Late progression rate',
                'rate',
                'gamma',
                late_progression_rate_horsburgh2010.estimate,
                late_progression_rate_horsburgh2010.interval, [],
                ['late_latent', 'active'])

timeperiod_activeuntreated \
    = Parameter('timeperiod_activeuntreated',
                'Untreated duration',
                'timeperiod',
                'gamma',
                untreated_duration_tiemersma2011.estimate,
                untreated_duration_tiemersma2011.interval, [],
                ['active disease (if not detected)'])

proportion_casefatality_active_untreated_smearpos \
    = Parameter('proportion_casefatality_active_untreated_smearpos',
                'Untreated case fatality smear-positive disease',
                'proportion',
                'beta_full_range',
                untreated_casefatality_smearpos_tiemersma2011.estimate,
                untreated_casefatality_smearpos_tiemersma2011.interval, [],
                ['death', 'death and spontaneous recovery (if not detected)'])

proportion_casefatality_active_untreated_smearneg \
    = Parameter('proportion_casefatality_active_untreated_smearneg',
                'Untreated case fatality smear-negative disease',
                'proportion',
                'beta_full_range',
                untreated_casefatality_smearneg_tiemersma2011.estimate,
                untreated_casefatality_smearneg_tiemersma2011.interval, [],
                ['death', 'death and spontaneous recovery (if not detected)'])

multiplier_relative_fitness_mdr \
    = Parameter('multiplier_relative_fitness_mdr',
                'Relative fitness',
                'multiplier',
                'beta_full_range',
                0.6, [0.25], [],
                ['force of infection'])

multiplier_bcg_protection \
    = Parameter('mutliplier_bcg_protection',
                'BCG protection',
                'multiplier',
                'beta_full_range',
                bcg_protection_colditz.estimate,
                bcg_protection_colditz.interval, [],
                ['force of infection parameter'])

timeperiod_treatment_ds \
    = Parameter('timeperiod_treatment_ds',
                'Time under treatment for DS-TB',
                'timeperiod',
                'normal_truncated',
                treatment_duration_ds_who2011.estimate,
                treatment_duration_ds_who2011.interval,
                [treatment_duration_ds_who2011.estimate, 1e3],
                ['time under treatment for DS-TB (may span multiple ' +
                 'compartments)'])
timeperiod_treatment_mdr \
    = Parameter('timeperiod_treatment_mdr',
                'Time under treatment for MDR-TB',
                'timeperiod',
                'normal_truncated',
                treatment_duration_mdr_who2011.estimate,
                treatment_duration_mdr_who2011.interval,
                [treatment_duration_mdr_who2011.estimate, 1e3],
                ['time under treatment for MDR-TB (may span multiple ' +
                 'compartments)'])

multiplier_force_smearneg \
    = Parameter('multiplier_relative_infectiousness_smearneg',
                'Relative infectiouness of smear negative disease',
                'multiplier',
                'beta_full_range',
                smearneg_transmission_tostmann2008.estimate,
                smearneg_transmission_tostmann2008.interval, [],
                ['force of infection parameter'])

mutlipler_force_treatment_ds \
    = Parameter('multipler_relative_infectiousness_treatment_ds',
                'Relative infectiousness of patients under treatment ' +
                '(DS-TB)',
                'multiplier',
                'beta_full_range',
                undertreatment_transmission_ds_dhamardhikari2014.estimate,
                undertreatment_transmission_ds_dhamardhikari2014.interval, [],
                ['force of infection parameter'])

mutlipler_force_treatment_mdr \
    = Parameter('multipler_relative_infectiousness_treatment_mdr',
                'Relative infectiousness of patients under treatment ' +
                '(MDR-TB)',
                'multiplier',
                'beta_full_range',
                undertreatment_transmission_mdr_dhamardhikari2014.estimate,
                undertreatment_transmission_mdr_dhamardhikari2014.interval, [],
                ['force of infection parameter'])

proportion_amplification_tomdr \
    = Parameter('proportion_amplification_tomdr',
                'Proportion of DS-TB defaults amplifying to new MDR-TB',
                'proportion',
                'beta_full_range',
                amplificiation_tomdr_bonnet2011.estimate,
                amplificiation_tomdr_bonnet2011.interval, [],
                ['defaults to active MDR-TB', 'all defaults from DS regimens'])
                
multiplier_negativeoutcome_retreatment_ds \
    = Parameter('multiplier_negativeoutcome_retreatment_ds',
                'Relative greater proportion with poor treatment outcomes ' +
                'for retreatment cases compared to new in DS-TB',
                'multiplier',
                'gamma',
                negativeoutcome_retreatment_ds_espinal2000.estimate,
                negativeoutcome_retreatment_ds_espinal2000.interval, [],
                ['defaults and deaths in retreatment DS-TB patients'])

proportion_treatmentsuccess_inappropriate_new_mdr \
    = Parameter('proportion_treatmentsuccess_inappropriate_new_mdr',
                'Treatment success for inappropriately treated new MDR cases',
                'proportion',
                'beta_full_range',
                inappropriatetreatment_new_mdr_espinal2000.estimate,
                inappropriatetreatment_new_mdr_espinal2000.interval, [],
                ['successful outcome', 'all new MDR patients treated with ' +
                'inappropriate first line regimen'])

proportion_treatmentsuccess_inappropriate_retreatment_mdr \
    = Parameter('proportion_treatmentsuccess_inappropriate_retreatment_mdr',
                'Treatment success for inappropriately treated retreatment ' +
                'MDR cases',
                'proportion',
                'beta_full_range',
                inappropriatetreatment_retreatment_mdr_espinal2000.estimate,
                inappropriatetreatment_retreatment_mdr_espinal2000.interval, [],
                ['successful outcome', 'all new MDR patients treated with ' +
                'inappropriate first line regimen'])

tb_n_contact \
    = Parameter('tb_n_contact',
                'Effective contact rate',
                'multiplier',
                'gamma',
                25.,
                [20.], [],
                ['rate_force'])

timeperiod_infect_ontreatment \
    = Parameter('timeperiod_infect_ontreatment',
                'Duration of infectiousness while on treatment',
                'timeperiod',
                'gamma',
                2. / 52.,
                [], [],
                ['Time in treatment_infect compartments before moving to treatment_infect'])