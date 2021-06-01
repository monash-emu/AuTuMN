import os

import pandas as pd

from autumn.projects.tuberculosis.marshall_islands.outputs.posteriors import (
    format_parameter,
)
from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    make_output_directories,
)
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "tuberculosis",
    "regions",
    "marshall_islands",
    "outputs",
    "figures",
    "priors_table",
)


def main():
    make_output_directories(FIGURE_PATH)
    make_priors_table(PRIORS)


def make_priors_table(priors):
    names = []
    ranges = []

    for prior in priors:
        if "_dispersion_param" in prior["param_name"]:
            continue
        names.append(format_parameter(prior["param_name"]))
        par_range = prior["distri_params"]
        ranges.append(f"{par_range[0]} - {par_range[1]}")
    df = pd.DataFrame({"Parameter": names, "Range": ranges})
    filename = os.path.join(FIGURE_PATH, "priors.csv")
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
