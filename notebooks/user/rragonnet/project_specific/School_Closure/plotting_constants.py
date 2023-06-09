from autumn.settings.folders import BASE_PATH, PROJECTS_PATH
from matplotlib import pyplot as plt
import os
import yaml

SCHOOL_PROJECT_NOTEBOOK_PATH = os.path.join(BASE_PATH, "notebooks", "user", "rragonnet", "project_specific", "School_Closure")
FIGURE_WIDTH = 8  # inches 
RESOLUTION = 300  # dpi

def set_up_style():
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "Times New Roman"


school_country_source = os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")
school_country_dict = yaml.load(open(school_country_source), Loader=yaml.UnsafeLoader)
INCLUDED_COUNTRIES = school_country_dict['all']
