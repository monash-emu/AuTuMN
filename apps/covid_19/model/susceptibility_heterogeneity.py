import scipy.special as special
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


def get_gamma(coeff_var: float):
    """
    Use function described in caption to Extended Data Fig 1 of Aguas et al pre-print to produce gamma distribution
    from coefficient of variation and independent variable.

    :param coeff_var:
    Independent variable
    :return: callable
    Function that provides the gamma distribution from the coefficient of variation
    """

    recip_cv_2 = coeff_var ** -2.

    def gamma_func(x_value: float):
        return x_value ** (recip_cv_2 - 1.) * \
               np.exp(-x_value * recip_cv_2) / \
               special.gamma(recip_cv_2) / \
               coeff_var ** (2. * recip_cv_2)

    return gamma_func


def numeric_integrate_gamma(coeff_var: float, lower_terminal: float, upper_terminal: float):
    """
    Numerically integrate previous function over a requested range
    :param coeff_var:
    Coefficient of variation or CV
    :param lower_terminal:
    Lower terminal value for numeric integration
    :param upper_terminal:
    Upper terminal value for numeric integration
    :return:
    Area under curve between lower and upper terminal
    """

    return integrate.quad(get_gamma(coeff_var), lower_terminal, upper_terminal)[0]


def get_bin_heights(plot_upper_limit, n_bins, coeff):
    """

    Note lower limit of zero is assumed

    :param plot_upper_limit:
    :param n_bins:
    :param coeff:
    :return:
    """

    lower_terminal, lower_terminals, upper_terminals, heights = \
        0., [], [], []
    bin_width = \
        plot_upper_limit / n_bins
    for i_bin in range(n_bins):
        lower_terminals.append(
            lower_terminal
        )
        upper_terminals.append(
            lower_terminal + bin_width
        )
        heights.append(
            numeric_integrate_gamma(
                coeff,
                lower_terminals[-1],
                upper_terminals[-1])
        )
        lower_terminal = \
            upper_terminals[-1]
    normalised_heights = [
        i_height / np.average(heights) for i_height in heights
    ]
    return lower_terminals, upper_terminals, normalised_heights, bin_width


def produce_gomes_exfig1(coeffs: list, add_hist=False, n_bins=3, x_values=50, plot_upper_limit=3.):
    """
    Produce figure equivalent to Extended Data Fig 1 of Aguas et al pre-print
    :return:
    """

    x_values = np.linspace(0.1, plot_upper_limit, x_values)  # Can't go to zero under some coeffs
    gamma_plot = plt.figure()
    axis = gamma_plot.add_subplot(111)
    for coeff in coeffs:
        gamma_func = get_gamma(coeff)

        # Line graph of the raw function
        y_values = [gamma_func(i) for i in x_values]
        axis.plot(x_values, y_values, color="k")

        # Numeric integration over parts of the function domain
        if add_hist:
            lower_terminals, _, normalised_heights, bin_width = get_bin_heights(plot_upper_limit, n_bins, coeff)
            axis.bar(
                lower_terminals, 
                normalised_heights, 
                width=bin_width, 
                align="edge"
            )

    return gamma_plot


produce_gomes_exfig1(coeffs=[0.5, 1., 2.], x_values=100, add_hist=False).savefig("gomes_exfig1.jpg")
