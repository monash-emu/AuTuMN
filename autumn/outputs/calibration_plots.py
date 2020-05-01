import os
from matplotlib import pyplot as plt
from scipy import stats
from numpy import linspace

from autumn.calibration.utils import calculate_prior, raise_error_unsupported_prior

plt.style.use('ggplot')


def plot_all_priors(priors, directory):
    """
    Make graphs to display prior distributions used in calibration
    :param priors: list of dictionaries
    :param directory: path to directory
    """
    print("Plotting prior distributions")
    path = os.path.join(directory, 'prior_plots')
    os.makedirs(path)
    for i, prior_dict in enumerate(priors):
        fig, ax = plt.subplots()
        x_range = workout_plot_x_range(prior_dict)
        x_values = linspace(x_range[0], x_range[1], num=1000)
        y_values = [calculate_prior(prior_dict, x, log=False) for x in x_values]
        zeros = [0. for i in x_values]
        plt.fill_between(x_values, y_values, zeros, color='cornflowerblue')
        plt.xlabel(prior_dict['param_name'])
        plt.ylabel('prior PDF')

        # place a text box in upper left corner to indicate the prior details
        props = dict(boxstyle='round', facecolor='dimgray', alpha=0.5)
        textstr = prior_dict['distribution'] + "\n(" + str(round(float(prior_dict['distri_params'][0]), 3)) + \
                  ", " + str(round(float(prior_dict['distri_params'][1]), 3))\
                  + ")"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        filename = os.path.join(path, prior_dict['param_name'] + ".png")
        plt.savefig(filename)


def workout_plot_x_range(prior_dict):
    if prior_dict["distribution"] == "uniform":
        x_range = prior_dict["distri_params"]
    elif prior_dict["distribution"] == "beta":
        a = prior_dict["distri_params"][0]
        b = prior_dict["distri_params"][1]
        x_range = stats.beta.ppf([.005, .995], a, b)
    elif prior_dict["distribution"] == "gamma":
        shape = prior_dict["distri_params"][0]
        scale = prior_dict["distri_params"][1]
        x_range = stats.gamma.ppf([.005, .995], shape, 0., scale)
    else:
        raise_error_unsupported_prior(prior_dict["distribution"])

    return x_range
