import scipy.special as special
import scipy.integrate as integrate
from scipy.stats import gamma as gamma_dist
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def get_gamma_data(tail_start_point, n_bins, coeff):
    """
    Compute a discretised version of a gamma distribution with mean 1 and a given coefficient of variation (sd/mean).
    The discretisation involves numerical solving to ensure the coefficient of variation is preserved. However, the mean
    of the discrete ditribution may be different to 1.
    :param tail_start_point: the last lower terminal
    :param n_bins: number of bins (including tail)
    :param coeff: requested coefficient of variation
    :return:
    """
    # First address a few singular scenarios
    if n_bins == 1 and coeff > 0.:
        raise ValueError("Cannot compute a positive coefficient of variation using a single bin")
    elif n_bins > 1 and coeff == 0.:
        raise ValueError("Cannot compute a null coefficient of variation using multiple bins")
    elif n_bins == 1:
        return [0.], [float("inf")], 1., 1., float("inf")

    n_finite_bins = n_bins - 1
    # Prelims
    lower_terminal, lower_terminals, upper_terminals, heights = \
        0., [], [], []
    bin_width = \
        tail_start_point / n_finite_bins

    for i_bin in range(n_finite_bins):

        # Record the upper and lower terminals
        lower_terminals.append(
            lower_terminal
        )
        upper_terminals.append(
            lower_terminal + bin_width
        )

        gamma_distri = gamma_dist(coeff ** -2., scale=coeff ** 2.)
        heights.append(gamma_distri.cdf(upper_terminals[-1]) - gamma_distri.cdf(lower_terminals[-1]))

        # Move to the next value
        lower_terminal = \
            upper_terminals[-1]

    # the last height is the remaining area under the curve (tail)
    heights.append(1. - sum(heights))

    # Find mid-points as the representative values
    mid_points = \
        [(lower + upper) / 2. for lower, upper in zip(lower_terminals, upper_terminals)]

    # add the last representative point such that modelled_CV = input_CV
    mid_points.append(
        find_last_representative_point(mid_points, heights, coeff)
    )
    lower_terminals.append(upper_terminals[-1])
    upper_terminals.append(float('inf'))

    # Return everything just in case
    return lower_terminals, upper_terminals, mid_points, heights, bin_width


def find_last_representative_point(mid_points, heights, coeff):
    """
    Find the last representative value of the discretised gamma distribution such that the requested coefficient of
    variation is presered.
    """

    def coeff_bias(last_point):
        values = mid_points + [last_point]
        mean = sum([prop * val for prop, val in zip(heights, values)])
        variance = sum([prop*(val - mean)**2 for prop, val in zip(heights, values)])
        modelled_cv = sqrt(variance) / mean
        return modelled_cv - coeff

    solution = optimize.root(coeff_bias, x0=mid_points[-1])
    best_last_point = solution.x[0]

    assert best_last_point > mid_points[-1]

    return best_last_point


def check_modelled_susc_cv(values, props, input_cv):
    """
    Check that the modelled coefficient of variation is close to the requested one.
    """
    mean = sum([prop * val for prop, val in zip(props, values)])
    variance = sum([prop*(val - mean)**2 for prop, val in zip(props, values)])
    modelled_cv = sqrt(variance) / mean
    relative_error = abs(modelled_cv - input_cv) / input_cv
    assert relative_error <= .1, "The modelled CV is not close enough to the input CV."


def produce_gomes_exfig1(coeffs: list, add_hist=False, n_bins=3, x_values=50, plot_upper_limit=3.):
    """
    Produce figure equivalent to Extended Data Fig 1 of Aguas et al pre-print as a check
    """

    # Prelims
    x_values = np.linspace(0.1, plot_upper_limit, x_values)  # Can't go to zero under some coeffs
    gamma_plot = plt.figure()
    axis = gamma_plot.add_subplot(111)

    # For each requested coefficient of variation
    for coeff in coeffs:
        gamma_distri = gamma_dist(coeff ** -2., scale=coeff ** 2.)

        # Line graph of the raw function
        y_values = [gamma_distri.pdf(i) for i in x_values]
        axis.plot(x_values, y_values, color="k")

        # Numeric integration over parts of the function domain
        if add_hist:
            lower_terminals, _, _, normalised_heights, bin_width = \
                get_gamma_data(
                    plot_upper_limit,
                    n_bins,
                    coeff
                )
            axis.bar(
                lower_terminals,
                normalised_heights,
                width=bin_width,
                align="edge"
            )

    # Return the figure
    return gamma_plot


# if __name__ == '__main__':
    # produce_gomes_exfig1(coeffs=[0.5, 1., 2.], x_values=100, n_bins=10, add_hist=False).savefig("gomes_exfig1.jpg")



    # lower_terminals, upper_terminals, mid_points, normalised_heights, bin_width = get_gamma_data(3., 10, 0.5)
    # print(f"lower terminals: {lower_terminals}")
    # print(f"upper terminals: {upper_terminals}")
    # print(f"mid-points: {mid_points}")
    # print(f"normalised heights: {normalised_heights}")
    # print(f"bin width: {bin_width}")
