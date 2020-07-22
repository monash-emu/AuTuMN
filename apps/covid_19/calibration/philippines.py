from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.PHILIPPINES,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


PAR_PRIORS = [
    {
        "param_name": "contact_rate", 
        "distribution": "uniform", 
        "distri_params": [0.010, 0.045],
    },
    {
        "param_name": "start_time", 
        "distribution": "uniform", 
        "distri_params": [0.0, 40.0],
    },
    {
            "param_name": "tv_detection_b",  # shape parameter
            "distribution": "uniform",
            "distri_params": [0.05, 0.1],
    },
    {
            "param_name": "tv_detection_c",  # inflection point
            "distribution": "uniform",
            "distri_params": [70.0, 110.0],
    },
    {
            "param_name": "prop_detected_among_symptomatic",  # upper asymptote
            "distribution": "uniform",
            "distri_params": [0.10, 0.90],
    },
    # Add extra params for negative binomial likelihood
    {
        "param_name": "notifications_dispersion_param",
        "distribution": "uniform",
        "distri_params": [0.1, 5.0],
    },
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.0,
        "distri_ci": [4.4, 5.6],
    },
    {
        "param_name": "compartment_periods_calculated.total_infectious.total_period",
        "distribution": "gamma",
        "distri_mean": 7.0,
        "distri_ci": [4.5, 9.5],
    },
    # parameters to derive age-specific IFRs
    {
        "param_name": "ifr_double_exp_model_params.k",
        "distribution": "uniform",
        "distri_params": [6., 14.],
    },
    {
        "param_name": "ifr_double_exp_model_params.last_representative_age",
        "distribution": "uniform",
        "distri_params": [75., 85.],
    },
]

# notification data:
notification_times = [
      44,
      53,
      61,
      68,
      75,
      82,
      89,
      96,
     103,
     110,
     117,
     124,
     131,
     138,
     145,
     152,
     159,
     166,
     173,
     180,
     187,
]

notification_values = [
         4,
         2,
         8,
        93,
       436,
      1080,
      1766,
      1381,
      1652,
      1435,
      1735,
      1849,
      1430,
      1811,
      4036,
      4005,
      3587,
      4913,
      5990,
     10377,
      9215,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1., 1.,] 
    },
]
