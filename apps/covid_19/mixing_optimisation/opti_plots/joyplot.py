import joypy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import os
from autumn.db.database import Database
from autumn.tool_kit.uncertainty import (
    export_compartment_size,
    collect_iteration_weights,
    collect_all_mcmc_output_tables,
)
import yaml


def get_perc_recovered_by_age_and_time(calibration_output_path, country_name, burn_in=500):
    mcmc_tables, output_tables, derived_output_tables = collect_all_mcmc_output_tables(
        calibration_output_path
    )
    weights = collect_iteration_weights(mcmc_tables, burn_in)

    perc_recovered = {}
    for agegroup_index in range(3):  # range(16)
        agegroup = 5 * agegroup_index
        n_recovered = export_compartment_size(
            "recoveredXagegroup_" + str(agegroup),
            mcmc_tables,
            output_tables,
            derived_output_tables,
            weights,
        )

        # work out denominator
        popsizes = {key: [0] * len(list(n_recovered.values())[0]) for key in n_recovered.keys()}
        for comp in output_tables[0].columns:
            if "agegroup_" + str(agegroup) in comp:
                comp_sizes = export_compartment_size(
                    comp, mcmc_tables, output_tables, derived_output_tables, weights
                )
                for key in n_recovered:
                    popsizes[key] = [x + y for (x, y) in zip(popsizes[key], comp_sizes[key])]

        perc_recovered[agegroup] = {}
        for key in n_recovered:
            perc_recovered[agegroup][key] = [
                100 * x / y for (x, y) in zip(n_recovered[key], popsizes[key])
            ]

    file_path = os.path.join("dumped_dict.yml")
    with open(file_path, "w") as f:
        yaml.dump(perc_recovered, f)

    return perc_recovered


def format_perc_recovered_for_joyplot(perc_recovered):
    months = [{"March": 61.0}, {"April": 92.0}, {"May": 122.0}, {"June": 153.0}]
    n_sample = len(list(perc_recovered[0].values())[0])
    data = pd.DataFrame()
    for month_dict in months:
        data[list(month_dict.keys())[0]] = ""
    data["Age"] = ""

    i = 0
    for sample_index in range(n_sample):
        for agegroup in perc_recovered:
            month_values = []
            for month_dict in months:
                time = list(month_dict.values())[0]
                month_values.append(perc_recovered[agegroup][str(time)][sample_index])
            data.loc[i] = month_values + ["age_" + str(agegroup)]
            i += 1

    plt.figure(dpi=380)
    fig, axes = joypy.joyplot(data, by="Age", column=[list(d.keys())[0] for d in months])

    plt.savefig("figures/test.png")


def make_joyplot_figure(perc_recovered):
    pass


if __name__ == "__main__":
    path = "../../../../data/outputs/calibrate/covid_19/for_plot_test/test_mcmc"

    perc_reco = get_perc_recovered_by_age_and_time(path, "belgium", burn_in=0)

    print("perc calculated")

    with open("dumped_dict.yml", "r") as yaml_file:
        dumped = yaml.safe_load(yaml_file)
    perc_reco = dumped

    format_perc_recovered_for_joyplot(perc_reco)
    print()
