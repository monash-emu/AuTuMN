import datetime
from typing import List, Dict

import matplotlib.ticker as ticker
from matplotlib import colors

PLOT_TEXT_DICT = {
    "contact_rate": "infection risk per contact",
    "hospital_props_multiplier": "hospital risk multiplier",
    "compartment_periods.icu_early": "pre-ICU period",
    "testing_to_detection.assumed_cdr_parameter": "CDR at base testing rate",
    "microdistancing.parameters.max_effect": "max effect microdistancing",
    "icu_occupancy": "ICU beds occupied",
    # TB model parameters
    "start_population_size": "initial population size",
    "progression_multiplier": "progression multiplier",
    "time_variant_tb_screening_rate.shape": "screening profile (max gradient)",
    "time_variant_tb_screening_rate.inflection_time": "screening profile (inflection time), year",
    "time_variant_tb_screening_rate.end_asymptote": "screening profile (final rate), per year",
    "user_defined_stratifications.location.adjustments.detection.ebeye": "rel. screening rate (Ebeye)",
    "user_defined_stratifications.location.adjustments.detection.other": "rel. screening rate (Other Isl.)",
    "extra_params.rr_progression_diabetes": "rel. progression rate (diabetes)",
    "rr_infection_recovered": "RR infection (recovered)",
    "pt_efficacy": "PT efficacy",
    "infect_death_rate_dict.smear_positive": "TB mortality (smear-pos)",
    "infect_death_rate_dict.smear_negative": "TB mortality (smear-neg)",
    "self_recovery_rate_dict.smear_positive": "Self cure rate (smear-pos)",
    "self_recovery_rate_dict.smear_negative": "Self cure rate (smear-neg)",
    "proportion_seropositive": "seropositive percentage",
    "infection_deaths": "deaths",
    "incidence": "incident episodes per day",
    "accum_deaths": "cumulative deaths",
    "new_hospital_admissions": "new hospitalisations per day",
    "new_icu_admissions": "new ICU admissions per day",
    "hospital_occupancy": "hospital beds occupied",
    "sojourn.compartment_periods_calculated.exposed.total_period": "incubation period",
    "sojourn.compartment_periods_calculated.active.total_period": "duration active",
    "mobility.microdistancing.behaviour.parameters.max_effect": "max effect microdistancing",
    "mobility.microdistancing.behaviour_adjuster.parameters.sigma": "microdist max wane",
    "mobility.microdistancing.behaviour_adjuster.parameters.c": "microdist wane time",
    "clinical_stratification.props.hospital.multiplier": "hospitalisation adjuster",
    "clinical_stratification.icu_prop": "ICU proportion",
    "sojourn.compartment_periods.icu_early": "pre-ICU period",
    "other_locations": "other locations",
    "manila": "national capital region",
    "clinical_stratification.props.symptomatic.multiplier": "sympt prop adjuster",
    "clinical_stratification.non_sympt_infect_multiplier": "asympt infect multiplier",
    "infection_fatality.multiplier": "IFR adjuster",
    "cdr": "Proportion detected among symptomatic",
    "prevalence": "Prevalence of active disease",
    "prop_detected_traced": "Proportion traced among detected cases",
    "prop_contacts_with_detected_index": "Proportion of contacts whose index is detected",
    "prop_contacts_quarantined": "Proportion quarantined among all contacts",
    "prop_incidence_strain_delta": "Proportion of Delta variant in new cases",
    "contact_tracing.assumed_trace_prop": "traced prop high prevalence",
    "unvaccinated_prop": "proportion unvaccinated (all ages)",
    "one_dose_only_prop": "proportion received only one dose (all ages)",
    "vaccinated_prop": "proportion fully vaccinated (all ages)",
    "at_least_one_dose_prop": "proportion received at least one dose (all ages)",
}

ALPHAS = (1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05)
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
COLORS = (
    # Blues
    ["lightsteelblue", "cornflowerblue", "royalblue", "navy"],
    # Purples
    ["plum", "mediumorchid", "darkviolet", "rebeccapurple"],
    # Greens
    ["palegreen", "mediumspringgreen", "mediumseagreen", "darkgreen"],
    # Yellows
    ["lightgoldenrodyellow", "palegoldenrod", "gold", "darkgoldenrod"],
    # Orangey-browns
    ["papayawhip", "navajowhite", "burlywood", "saddlebrown"],
    # Cyans
    ["lightcyan", "paleturquoise", "darkcyan", "darkslategrey"],
    # Greys
    ["lightgrey", "darkgrey", "dimgrey", "black"],
    # Reds
    ["lightsalmon", "tomato", "indianred", "darkred"],
    # Dark greens
    ["mediumseagreen", "seagreen", "green", "darkgreen"],
)
REF_DATE = datetime.date(2019, 12, 31)


def get_plot_text_dict(
    param_string, capitalise_first_letter=False, remove_underscore=True, remove_dot=True, get_short_text=False,
):
    """
    Get standard text for use in plotting as title, y-label, etc.
    """

    text = PLOT_TEXT_DICT[param_string] if param_string in PLOT_TEXT_DICT else param_string
    if "end_asymptote" in param_string:
        text = text.replace("parameters.end_asymptote", "")
    if capitalise_first_letter:
        text = text[0].upper() + text[1:]
    if remove_underscore:
        text = text.replace("_", " ")
    if remove_dot:
        text = text.replace(".", " ")
    return text


def change_xaxis_to_date(axis, ref_date, date_str_format="%#d-%b-%Y", rotation=30):
    """
    Change the format of a numerically formatted x-axis to date.
    """

    def to_date(x_value, pos):
        date = ref_date + datetime.timedelta(days=int(x_value))
        return date.strftime(date_str_format)

    date_format = ticker.FuncFormatter(to_date)
    axis.xaxis.set_major_formatter(date_format)
    axis.xaxis.set_tick_params(rotation=rotation)


def add_vertical_lines_to_plot(axis, lines: Dict):
    """
    Add labelled vertical lines to the plot axis according to a dictionary with standard attributes.
    All attributes of the line and text are currently hard-coded.
    """

    for text, position in lines.items():

        # Add the line itself
        axis.axvline(x=position, linestyle="--", alpha=0.7)

        # Add the text to accompany it
        upper_ylim = axis.get_ylim()[1]
        axis.text(x=position, y=upper_ylim * 0.97, s=text, rotation=90, ha="right", va="top")


def add_horizontal_lines_to_plot(axis, lines: Dict):
    """
    Add labelled horizontal lines to the plot axis according to a dictionary with standard attributes.
    All attributes of the line and text are currently hard-coded.
    """

    for text, position in lines.items():

        # Add the line itself
        axis.axhline(y=position, linestyle="--", alpha=0.7)

        # Add the text to accompany it
        lower_xlim = axis.get_xlim()[0]
        axis.text(x=lower_xlim, y=position, s=text, ha="left", va="bottom")


def _apply_transparency(color_list: List[str], alphas: List[str]):
    """Make a list of colours transparent, based on a list of alphas"""

    # +++FIXME
    # This will fail if len(color_list) > len(alphas)
    # Should move to generative colours rather than fixed lists

    out_colors = []

    for i in range(len(color_list)):
        out_colors.append([])
        for j in range(len(color_list[i])):
            rgb_color = list(colors.colorConverter.to_rgb(color_list[i][j]))
            out_colors[i].append(rgb_color + [alphas[i]])

    return out_colors


def _plot_targets_to_axis(axis, values: List[float], times: List[int], on_uncertainty_plot=False):
    """
    Plot output value calibration targets as points on the axis.
    # TODO: add back ability to plot confidence interval
    x_vals = [time, time]
    axis.plot(x_vals, values[1:], "m", linewidth=1, color="red")
    axis.scatter(time, values[0], marker="o", color="red", s=30)
    axis.scatter(time, values[0], marker="o", color="white", s=10)
    """
    assert len(times) == len(values), "Targets have inconsistent length"
    # Plot a single point estimate
    if on_uncertainty_plot:
        axis.scatter(times, values, marker="o", color="black", s=10, zorder=999)
    else:
        axis.scatter(times, values, marker="o", color="red", s=30, zorder=999)
        axis.scatter(times, values, marker="o", color="white", s=10, zorder=999)


def split_mcmc_outputs_by_chain(mcmc_params, mcmc_tables):
    chain_ids = mcmc_params[0]["chain"].unique().tolist()
    mcmc_params_list, mcmc_tables_list = [], []
    for i_chain in chain_ids:
        mcmc_params_list.append(
            mcmc_params[0][mcmc_params[0]["chain"] == i_chain]
        )
        mcmc_tables_list.append(
            mcmc_tables[0][mcmc_tables[0]["chain"] == i_chain]
        )

    return mcmc_params_list, mcmc_tables_list
