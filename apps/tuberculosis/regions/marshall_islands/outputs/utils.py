import os
from matplotlib import pyplot

OUTPUT_TITLES = {
    "incidence": "TB incidence (/100,000/y)",
    "notifications": "Number of TB notifications",
    "mortality": "TB mortality (/100,000/y)",
    "percentage_latent": "LTBI prevalence (%)",
}

REGION_TITLES = {
    "majuro": "Majuro atoll",
    "ebeye": "Ebeye atoll",
    "all": "National level",
}


def get_format():
    pyplot.style.use("ggplot")
    pyplot.rcParams["font.family"] = "Times New Roman"


def save_figure(filename, figure_path):
    png_path = os.path.join(figure_path, f"{filename}.png")
    pdf_path = os.path.join(figure_path, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)
