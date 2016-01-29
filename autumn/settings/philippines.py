
from ..parameter import Parameter, Evidence

# EVIDENCE_____________________________________________________________________

bcg_coverage_unicef2014 \
    = Evidence('bcg_coverage_unicef2014',
               'Coverage of BCG vaccination',
               0.87, [0.76, 0.97],
               'BCG vaccination coverage at birth',
               '87% BCG coverage in 2014',
               'WHO-UNICEF estimates of BCG coverage, ' +
               'http://apps.who.int/immunization_monitoring/' +
               'globalsummary/timeseries/tswucoveragebcg.html')

# PARAMETERS___________________________________________________________________

bcg_coverage \
    = Parameter('program_prop_vac',
                'BCG coverage at birth',
                'proportion',
                'beta_full_range',
                bcg_coverage_unicef2014.estimate,
                bcg_coverage_unicef2014.interval, [],
                ['births_vac',
                 'births_vac + births_unvac'])

algorithm_sensitivity \
    = Parameter('algorithm_sensitivity',
                'Proportion of presenting cases correctly diagnosed',
                'proportion',
                'beta_full_range',
                0.85, [], [],  # Arbitrary at this point
                ['program_rate_detect', 'program_rate_detect + program_rate_miss'])

treatment_start_rate \
    = Parameter('treatment_start_rate',
                'Reciprocal of delay to treatment commencement after detection',
                'timeperiod',
                'gamma',
                26., [], [],  # Also arbitrary
                ['detect (by pulmonary_status)'])

treatment_start_rate \
    = Parameter('treatment_start_rate',
                'Reciprocal of delay to treatment commencement after detection',
                'timeperiod',
                'gamma',
                26., [], [],  # Also arbitrary
                ['detect (by pulmonary_status)'])

timeperiod_norepresentation \
    = Parameter('timeperiod_norepresentation',
                'Rate at which patients re-start presenting after failed diagnostic attempt',
                'timeperiod',
                'gamma',
                4., [], [],  # Guess what - arbitrary
                ['missed (by pulmonary status'])

proportion_default \
    = Parameter('proportion_default',
                'Proportion of patients defaulting from treatment ("default rate")',
                'proportion',
                'beta_full_range',
                0.1, [], [],  # Arbitrary for now
                ['Gets converted into flow in code within the __init__ method of the model object', ''])

proportion_death \
    = Parameter('proportion_death',
                'Proportion of patients defaulting from treatment ("death rate")',
                'proportion',
                'beta_full_range',
                0.1, [], [],  # Arbitrary for now
                ['Gets converted into flow in code within the __init__ method of the model object', ''])

program_timeperiod_delayto_treatment \
    = Parameter('program_timeperiod_delayto_treatment',
                'Time after diagnosis waiting for a regimen to become available',
                'timeperiod',
                'normal_truncated',
                2. / 52., [], [0., 1e3],  #Arbitrary
                ['Time in detect compartments'])

