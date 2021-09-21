import os

import pandas as pd

from autumn.projects.tuberculosis.marshall_islands.project import priors

from autumn.projects.tuberculosis.marshall_islands.outputs.posteriors import (
    format_parameter,
)
from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    make_output_directories,
)

PRIORS = priors


def main(output_path):
    figure_path = os.path.join(output_path, "priors_table")
    make_output_directories(figure_path)
    make_priors_table(PRIORS, figure_path)


def make_priors_table(priors, figure_path):
    names = []
    ranges = []

    for prior in priors:
        if "_dispersion_param" in prior.name:
            continue
        names.append(format_parameter(prior.name))
        par_range = prior.start, prior.end
        ranges.append(f"{par_range[0]} - {par_range[1]}")
    df = pd.DataFrame({"Parameter": names, "Range": ranges})
    filename = os.path.join(figure_path, "priors.csv")
    df.to_csv(filename, index=False)
