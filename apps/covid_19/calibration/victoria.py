from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import provide_default_calibration_params, add_standard_dispersion_parameter
from apps.covid_19.mixing_optimisation.utils import add_dispersion_param_prior_for_gaussian

def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.VICTORIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


case_times = [
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
]
case_counts = [
    1,
    0,
    0,
    0,
    2,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    3,
    0,
    0,
    2,
    0,
    0,
    1,
    0,
    0,
    1,
    1,
    3,
    3,
    3,
    6,
    9,
    13,
    8,
    14,
    23,
    27,
    29,
    28,
    51,
    67,
    61,
    56,
    55,
    54,
    54,
    111,
    84,
    52,
    96,
    51,
    68,
    49,
    30,
    20,
    23,
    33,
    21,
    16,
    13,
    24,
    3,
    13,
    10,
    8,
    2,
    1,
    17,
    9,
    1,
    7,
    2,
    1,
    6,
    3,
    3,
    1,
    2,
    3,
    7,
    3,
    7,
    13,
    22,
    17,
    17,
    14,
    14,
    11,
    10,
    7,
    17,
    7,
    9,
    21,
    11,
    7,
    8,
    7,
    8,
    4,
    12,
    10,
    2,
    2,
    5,
    8,
    10,
    7,
    11,
    6,
    4,
    10,
    7,
    8,
    3,
    0,
    4,
    2,
    0,
    4,
    8,
    4,
    8,
    9,
    12,
    9,
    21,
    18,
    13,
    25,
    19,
    16,
    17,
    20,
    33,
    30,
    41,
    49,
    75,
    64,
    73,
    77,
    66,
    108,
    74,
    127,
    191,
    134,
    165,
    288,
    216,
    273,
    177,
    270,
    238,
    317,
    428,
    217,
    363,
    275,
    374,
    484,
    403,
    300,
    357,
    459,
    532,
    384,
    295,
    723,
    627,
    397,
    671,
]

icu_times = [
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
]
icu_counts = [
    6,
    7,
    7,
    10,
    11,
    11,
    13,
    12,
    13,
    13,
    15,
    14,
    14,
    15,
    18,
    17,
    13,
    12,
    10,
    11,
    12,
    13,
    10,
    11,
    11,
    10,
    11,
    11,
    9,
    9,
    7,
    7,
    7,
    6,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    4,
    6,
    6,
    7,
    7,
    7,
    5,
    5,
    5,
    5,
    5,
    3,
    3,
    3,
    3,
    4,
    3,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    2,
    1,
    1,
    1,
    2,
    3,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    3,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    4,
    6,
    3,
    3,
    5,
    9,
    7,
    9,
    12,
    15,
    16,
    17,
    26,
    27,
    29,
    31,
    25,
    28,
    31,
    36,
    40,
    40,
    41,
    42,
    42,
    44,
    42,
    41,
    34,
    36,
    41,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": case_times,
        "values": case_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(case_times) - 6)) + [250.] * 7,
    },
    {
        "output_key": "icu_occupancy",
        "years": icu_times,
        "values": icu_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(icu_times) - 6)) + [250.] * 7,
    },
]

PAR_PRIORS = provide_default_calibration_params(["start_time"])
# PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
# PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")

PAR_PRIORS += [
    # Programmatic parameters
    {
        "param_name": "seasonal_force",
        "distribution": "uniform",
        "distri_params": [0., 0.4],
    },
    # {
    #     "param_name": "time_variant_detection.end_value",
    #     "distribution": "beta",
    #     "distri_mean": 0.85,
    #     "distri_ci": [0.6, 0.9],
    # },
    {
        "param_name": "testing_to_detection.maximum_detection",
        "distribution": "uniform",
        "distri_params": [0.6, 0.95],
    },
    {
        "param_name": "compartment_periods.icu_early",
        "distribution": "gamma",
        "distri_mean": 12.,
        "distri_ci": [4., 20.],
    },
    {
        "param_name": "compartment_periods.icu_late",
        "distribution": "uniform",
        "distri_params": [5., 15.],
    },
    {
        "param_name": "icu_prop",
        "distribution": "uniform",
        "distri_params": [0.12, 0.22],
    },
    {
        "param_name": "symptomatic_props_multiplier",
        "distribution": "uniform",
        "distri_params": [0.5, 2.]
    },
    {
        "param_name": "testing_to_detection.shape_parameter",
        "distribution": "uniform",
        "distri_params": [-6, -3]
    }
]

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS, {})
