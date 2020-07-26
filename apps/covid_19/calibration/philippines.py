from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import \
    provide_default_calibration_params, add_standard_dispersion_parameter, add_case_detection_params_philippines, \
    assign_trailing_weights_to_halves


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


# Notification data:
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
     194,
]

notification_values = [
        1,
        1,
        2,
       10,
       68,
      151,
      231,
      192,
      250,
      218,
      237,
      275,
      210,
      255,
      515,
      617,
      503,
      703,
      832,
     1406,
     1409,
     1734,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": assign_trailing_weights_to_halves(5, notification_times),
    },
]

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
PAR_PRIORS = add_case_detection_params_philippines(PAR_PRIORS)
