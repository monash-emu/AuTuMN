import autumn.projects.tuberculosis.kiribati.outputs.calibration as cal
import autumn.projects.tuberculosis.kiribati.outputs.counterfactual as ctf
import autumn.projects.tuberculosis.kiribati.outputs.diabetes as dia
import autumn.projects.tuberculosis.kiribati.outputs.elimination as elm
import autumn.projects.tuberculosis.kiribati.outputs.get_output_numbers as gon
import autumn.projects.tuberculosis.kiribati.outputs.posteriors as pos
import autumn.projects.tuberculosis.kiribati.outputs.priors as pri
from autumn.projects.tuberculosis.kiribati.outputs.utils import get_format

import os
from autumn.settings import BASE_PATH

DATA_PATH = os.path.join(
    BASE_PATH, "autumn", "projects", "tuberculosis", "kiribati", "outputs", "pbi_databases"
)
OUTPUT_PATH = os.path.join(
    BASE_PATH, "autumn", "projects", "tuberculosis", "kiribati", "outputs", "all_outputs"
)


def make_all_rmi_plots(analysis="main"):
    data_path = os.path.join(DATA_PATH, analysis)
    output_path = os.path.join(OUTPUT_PATH, analysis)

    get_format()

    # Print outputs as numbers
    gon.main(data_path, output_path)

    # calibration outputs
    cal.main(data_path, output_path)

    # prior table
    pri.main(output_path)

    # posterior table
    # pos.main(data_path, output_path)

    # counterfactual outputs
    ctf.main(data_path, output_path)

    # elimination outputs
    elm.main(data_path, output_path)

    # diabetes plot
    dia.main(data_path, output_path)


for analysis in ["main", "rmi_bcg_mortality", "rmi_constant_cdr", "rmi_more_intermixing"]:
    print(f"Plotting outputs for analysis {analysis}")
    make_all_rmi_plots(analysis)
    for _ in range(5):
        print()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

