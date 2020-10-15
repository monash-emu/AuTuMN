import matplotlib.ticker as ticker
import datetime


PLOT_TEXT_DICT = {
    "contact_rate": "infection risk per contact",
    "compartment_periods_calculated.exposed.total_period": "incubation period",
    "compartment_periods_calculated.active.total_period": "duration active",
    "hospital_props_multiplier": "hospital risk multiplier",
    "compartment_periods.icu_early": "pre-ICU period",
    "icu_prop": "ICU proportion",
    "testing_to_detection.assumed_cdr_parameter": "CDR at base testing rate",
    "microdistancing.parameters.max_effect": "max effect microdistancing",
    "icu_occupancy": "ICU occupancy",
    # TB model parameters
    "start_population_size": "initial population size",
    "late_reactivation_multiplier": "late reactivation multiplier",
    "time_variant_tb_screening_rate.maximum_gradient": "screening profile (shape)",
    "time_variant_tb_screening_rate.max_change_time": "screening profile (inflection)",
    "time_variant_tb_screening_rate.end_value": "screening profile (final rate)",
    "user_defined_stratifications.location.adjustments.detection_rate.ebeye": "rel. screening rate (Ebeye)",
    "user_defined_stratifications.location.adjustments.detection_rate.other": "rel. screening rate (Other Isl.)",
    "extra_params.rr_progression_diabetes": "rel. progression rate (diabetes)",
    "rr_infection_recovered": "RR infection (recovered)",
    "pt_efficacy": "PT efficacy",
    "infect_death_rate_dict.smear_positive": "TB mortality (smear-pos)",
    "infect_death_rate_dict.smear_negative": "TB mortality (smear-neg)",
    "self_recovery_rate_dict.smear_positive": "Self cure rate (smear-pos)",
    "self_recovery_rate_dict.smear_negative": "Self cure rate (smear-neg)",
    "proportion_seropositive": "seropositive percentage",
    "infection_deaths": "deaths per day",
    "notifications": "notifications per day",
    "incidence": "incident episodes per day",
}


def get_plot_text_dict(param_string, capitalise_first_letter=False):
    text = PLOT_TEXT_DICT[param_string] if param_string in PLOT_TEXT_DICT else param_string
    if capitalise_first_letter:
        text = text[0].upper() + text[1:]
    return text


def change_xaxis_to_date(axis, ref_date, date_str_format="%#d-%b", rotation=30):
    """
    Change the format of a numerically formatted x-axis to date.
    """

    def to_date(x_value, pos):
        date = ref_date + datetime.timedelta(days=x_value)
        return date.strftime(date_str_format)

    date_format = ticker.FuncFormatter(to_date)
    axis.xaxis.set_major_formatter(date_format)
    axis.xaxis.set_tick_params(rotation=rotation)
