import os

import pandas as pd

from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    make_output_directories,
)
from autumn.coreplots.utils import PLOT_TEXT_DICT


PARAMETER_NAMES_OVERRIDE = {
    "rel. screening rate (Ebeye)": "relative rate of passive TB screening in Ebeye (ref. Majuro)",
    "rel. screening rate (Other Isl.)": "relative rate of passive TB screening in other islands (ref. Majuro)",
    "rel. progression rate (diabetes)": "relative rate of TB progression for diabetic individuals",
    "RR infection (recovered)": "relative risk of infection for individuals with history of infection (ref. infection-naive)",
    "PT efficacy": "efficacy of preventive treatment",
    "TB mortality (smear-pos)": "TB mortality (smear-positive), per year",
    "TB mortality (smear-neg)": "TB mortality (smear-negative), per year",
    "Self cure rate (smear-pos)": "Self-cure rate (smear-positive), per year",
    "Self cure rate (smear-neg)": "Self-cure rate (smear-negative), per year",
    "rr_infection_latent": "relative risk of infection for individuals with latent infection (ref. infection-naive)",
    "awareness_raising.relative_screening_rate": "relative screening rate following ACF interventions (ref. before intervention)",
    "infection risk per contact": "transmission scaling factor",
}


def main(data_path, output_path):
    figure_path = os.path.join(output_path, "posterior_table")
    make_output_directories(figure_path)

    file_path = os.path.join(data_path, "parameter_posteriors", "posterior_centiles.csv")
    posterior_df = pd.read_csv(file_path, sep=",")
    posterior_df = posterior_df.rename(
        columns={
            "Unnamed: 0": "Parameter",
            "2.5": "2.5th percentile",
            "50.0": "Median",
            "97.5": "97.5th percentile",
        }
    )
    make_posterior_table(posterior_df, figure_path)


def format_parameter(parameter):
    if parameter in PLOT_TEXT_DICT:
        parameter_name = PLOT_TEXT_DICT[parameter]
    else:
        print(f"No translation found for parameter {parameter} ")
        parameter_name = parameter

    if parameter_name in PARAMETER_NAMES_OVERRIDE:
        return PARAMETER_NAMES_OVERRIDE[parameter_name]
    else:
        return parameter_name


def make_posterior_table(posterior_df, figure_path):

    posterior_df.Parameter = posterior_df.Parameter.apply(format_parameter)

    filename = "posterior_ranges.csv"
    file_path = os.path.join(figure_path, filename)
    posterior_df.to_csv(file_path)
