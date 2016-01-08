# -*- coding: utf-8 -*-

"""
Created on Sat Nov 28 12:01:25 2015

@author: James
"""

from parameter_estimation import Parameter, Evidence

# PROGRESSION PROPORTION FROM LATENCY
# Taken directly from early_progression_trauer2016
early_progression \
    = Parameter('Early progression proportion',
                'proportion',
                'beta_full_range',
                0.125, 0.015, 0,
                ['latent_early to active',
                 'latent_early to latent_late'])
early_progression_child \
    = Parameter('Early progression proportion',
                'proportion',
                'beta_full_range',
                0.42, 0.05, 0,
                ['latent_early to active',
                 'latent_early to latent_late'])
early_progression_adult \
    = Parameter('Early progression proportion',
                'proportion',
                'beta_full_range',
                0.024, 0.015, 0,
                ['latent_early to active',
                 'latent_early to latent_late'])

# Trauer early progression estimates
early_progression_trauer2016 \
    = Evidence('Early progression proportion', 0.125,
               'Early progression proportion, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_trauer2016 \
    = Evidence('Early progression proportion - children', 0.42,
               'Early progression proportion, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_trauer2016 \
    = Evidence('Early progression proportion - adults', 0.024,
               'Early progression proportion, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

# Sloot early progression estimates
early_progression_sloot2014 \
    = Evidence('Early progression proportion', 0.095,
               'Early progression proportion, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, approximately 9 to 10% of the 739 contacts ' +
               'with evidence of infection developed active TB in the early ' +
               'high risk period, which lasted for around two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_child_sloot2014 \
    = Evidence('Early progression proportion - children', 0.26,
               'Early progression proportion, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, approximately 9 to 10% of the 739 contacts ' +
               'with evidence of infection developed active TB in the early ' +
               'high risk period, which lasted for around two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_progression_adult_sloot2014 \
    = Evidence('Early progression proportion - adults', 0.07,
               'Early progression proportion, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, approximately 9 to 10% of the 739 contacts ' +
               'with evidence of infection developed active TB in the early ' +
               'high risk period, which lasted for around two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

# DURATION OF EARLY LATENCY
# Taken directly from early_progression_trauer2016
early_latent_duration \
    = Parameter('Time spent in early latency',
                'sojourn time',
                'gamma',
                5. / 12., 5. / 12. / 2., 0, # Range arbitrary
                ['reciprocal of early_latent flows - early progression ' +
                'and stabilisation'])
early_latent_duration_child \
    = Parameter('Time spent in early latency - children',
                'sojourn time',
                'gamma',
                147. / 365., 147. / 365. / 2., 0,
                ['reciprocal of early_latent flows - early progression ' +
                'and stabilisation'])
early_latent_duration_adult \
    = Parameter('Time spent in early latency - adults',
                'sojourn time',
                'gamma',
                227. / 365., 227. / 365. / 2., 0,
                ['reciprocal of early_latent flows - early progression ' +
                'and stabilisation'])

# Trauer early latency duration 
early_latent_duration_trauer2016 \
    = Evidence('Early latent duration', 5. / 12.,
               'Early latent duration, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')
early_latent_duration_child_trauer2016 \
    = Evidence('Early latent duration in children', 145. / 365.,
               'Early latent duration, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')
early_latent_duration_adult_trauer2016 \
    = Evidence('Early latent duration in adults', 227. / 365.,
               'Early latent duration, Trauer et al. 2016',
               'early_progression_trauer2016.pdf',
               'From the text of our Results (first paragraph).',
                'Trauer JM, Moyo N, Tay E-L, Dale K, Ragonnet R, ' +
                'McBryde ES, Denholm JT. ' +
                'Risk of Active Tuberculosis in the Five Years Following ' +
                'Infection . . . 15%' +
                'Chest. 2016;149(2):516-525.')

# Sloot early latency duration
early_latent_duration_sloot2014 \
    = Evidence('Early latent duration', 1.8,
               'Early latent duration, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_child_sloot2014 \
    = Evidence('Early latent duration in children', 0.6,
               'Early latent duration, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')
early_latent_duration_adult_sloot2014 \
    = Evidence('Early latent duration in adults', 1.8,
               'Early latent duration, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

# PROGRESSION RATE LATE LATENCY
late_progression \
    = Parameter('Late progression rate',
                'rate',
                'gamma',
                0.058 / 100., (0.089 - 0.038) / 4. / 100., 0,
                ['late_latent to active rate'])

late_progression_rate_horsburgh2010 \
    = Evidence('Late progression rate', 0.058 / 100.,
               'Late progression rate, Horsburgh et al. 2010',
               'late_progression_horsburgh2010.pdf',
               'Main finding is rate of reactivation of 0.04 per year',
               'Community based estimate of rate of reactivation from ' +
               'the US.')

# UNTREATED DISEASE DURATION
untreated_duration \
    = Parameter('Untreated duration',
                'sojourn time',
                'gamma',
                3., 1., 0, # Spread arbitrary, WHO use 1-4 in their calculations though
                ['reciprocal of active to late_latent or death total flows'])

untreated_duration_tiemersma \
    = Evidence('Untreated duration', 3,
               'Untreated duration, Tiemersma et al. 2011',
               'untreated_duration_tiemersma2011.pdf',
               'Estimate from pre-chemotherapy literature of three years ' +
               'untreated', 'No confidence interval around this estimate')

# RELATIVE TRANSMISSIBILITY MDR-TB
relative_fitness_mdr \
    = Parameter('Relative fitness',
                'multiplier',
                'beta_full_range',
                0.6, 0.25, 0,
                ['multiplier for force of infection parameter'])

# RELATIVE RISK OF TB AFTER BCG VACCINATION
bcg_protection \
    = Parameter('BCG protection',
                'multiplier',
                'beta_full_range',
                0.49, (0.7 - 0.34) / 4., 0,
                ['multiplier for force of infection parameter'])

bcg_protection_colditz \
    = Evidence('BCG protection', 0.49,
               'Relative risk of TB, Colditz et al. 1994',
               'bcg_protection_colditz1994.pdf',
               'Estimate of relative risk of TB in BCG vaccinated',
               'Condfidence interval given for RR estimate')
