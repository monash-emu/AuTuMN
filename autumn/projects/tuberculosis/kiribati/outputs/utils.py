import os

from matplotlib import pyplot

from autumn.settings import BASE_PATH

OUTPUT_TITLES = {
    "incidence": "TB incidence (/100,000/y)",
    "notifications": "Number of TB notifications",
    "mortality": "TB mortality (/100,000/y)",
    "percentage_latent": "LTBI prevalence (%)",
    "prevalence_infectiousXlocation_starawa": "TB prevalence in South Tarawa (/100,000/y)",
    "prevalence_infectiousXlocation_starawa": "TB prevalence in South Tarawa (/100,000/y)",
    "percentage_latentXlocation_starawa": "LTBI prevalence in South Tarawa (%)",
    "notificationsXlocation_starawa": "Number of TB notifications in South Tarawa",
    "notificationsXlocation_ebeye": "Number of TB notifications in Ebeye",
    "population_size": "Population size",
    "screening_rate": "Rate of passive TB screening (/year)",
    "cumulative_pt": "Cumulative number of PTs",
    "cumulative_pt_sae": "Cummulative serious adverse effects from PT",
}

REGION_TITLES = {
    "starawa": "South Tarawa",
    "other": "Other Islands",
    "all": "National level",
}

INTERVENTION_TITLES = {
    "ACF": "Community-wide ACF",
    "ACF_LTBI": "Community-wide ACF and\npreventive treatment",
    "hh_pt": "Preventive treatment\nfor all contacts",
}


def make_output_directories(output_path):
    base_figure_path = os.path.join(
        BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "figures"
    )
    for dir_path in [base_figure_path, output_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def get_format():
    pyplot.style.use("ggplot")
    pyplot.rcParams["font.family"] = "Times New Roman"


def save_figure(filename, figure_path):
    png_path = os.path.join(figure_path, f"{filename}.png")
    png_path_lowres = os.path.join(figure_path, f"{filename}_low_res.png")
    pdf_path = os.path.join(figure_path, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(png_path_lowres, dpi=200)
    pyplot.savefig(pdf_path)
