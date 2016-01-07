# -*- coding: utf-8 -*-

"""
Created on Sat Nov 28 12:01:25 2015

@author: James
"""

from parameter_estimation import Parameter, Evidence

# Estimate of early progression rate from Sloot, et al.
early_progression \
    = Parameter('Early progression proportion',
                'proportion',
                'beta_full_range',
                0.095, 0.02, 0,
                ['latent_early to active',
                 'latent_early to latent_late'])

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

# Estimate of time spent in early latency from Sloot, et al.
early_latent_duration \
    = Parameter('Time spend in early latency',
                'sojourn time',
                'gamma',
                2, 1.5, 0,
                ['reciprocal of early_latent flows - early progression '
                + 'and stabilisation'])

early_latent_duration_sloot2014 \
    = Evidence('Early latent duration', 2,
               'Early latent duration, Sloot et al. 2014',
               'early_progression_sloot2014.pdf',
               'From Figure 2, high risk period lasts about two years.',
               'Sloot R, Schim van der Loeff MF, Kouw PM, Borgdorff MW. ' +
               'Risk of Tuberculosis after Recent Exposure. A 10-Year ' +
               'Follow-up Study of Contacts in Amsterdam. Am J Respir ' +
               'Crit Care Med. 2014;190(9):1044-1052.')

# Estimate of late latency progression rate from Horsburgh, et al.
# Spread calculation possibly incorrect, thought the gamma scale parameter was
# based on variance, rather than standard deviation.
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

# Estimate of untreated duration from Tiemersma, et al.
# Note that spread is made up
untreated_duration \
    = Parameter('Untreated duration',
                'sojourn time',
                'gamma',
                3., 1., 0,
                ['reciprocal of active to late_latent or death total flows'])

untreated_duration_tiemersma \
    = Evidence('Untreated duration', 3,
               'Untreated duration, Tiemersma et al. 2011',
               'untreated_duration_tiemersma2011.pdf',
               'Estimate from pre-chemotherapy literature of three years ' +
               'untreated', 'No confidence interval around this estimate')

# Estimate of relative transmissibility of MDR-TB - no-one really knows
relative_fitness_mdr \
    = Parameter('Relative fitness',
                'multiplier',
                'beta_full_range',
                0.6, 0.25, 0,
                ['multiplier for force of infection parameter'])

# Estimate of relative risk of infection after BCG vaccination
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

bcg_protection_colditz.open_pdf()