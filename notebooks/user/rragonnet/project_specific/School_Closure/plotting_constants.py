from autumn.settings.folders import BASE_PATH
from matplotlib import pyplot as plt
import os

SCHOOL_PROJECT_NOTEBOOK_PATH = os.path.join(BASE_PATH, "notebooks", "user", "rragonnet", "project_specific", "School_Closure")
FIGURE_WIDTH = 8  # inches 
RESOLUTION = 300  # dpi

def set_up_style():
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "Times New Roman"