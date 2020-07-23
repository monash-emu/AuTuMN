from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import \
    provide_default_calibration_params, add_standard_dispersion_parameter, add_case_detection_params_philippines


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

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
PAR_PRIORS = add_case_detection_params_philippines(PAR_PRIORS)
