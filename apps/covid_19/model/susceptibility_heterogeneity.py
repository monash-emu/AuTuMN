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


def numeric_integrate_gamma(gamma_function, lower_terminal: float, upper_terminal: float):
    """
    Numerically integrate previous function over a requested range

    :param lower_terminal:
    Lower terminal value for numeric integration
    :param upper_terminal:
    Upper terminal value for numeric integration
    :return:
    Area under curve between lower and upper terminal
    """

    return integrate.quad(gamma_function, lower_terminal, upper_terminal)[0]


def get_gamma_data(domain_upper_limit, n_bins, coeff):
    """

    Note lower limit of zero is assumed

    :param domain_upper_limit:
    :param n_bins:
    :param coeff:
    :return:
    """

    # Prelims
    lower_terminal, lower_terminals, upper_terminals, heights = \
        0., [], [], []
    bin_width = \
        domain_upper_limit / n_bins

    # Get the gamma function based on the coefficient needed
    gamma_function = get_gamma(coeff)

    for i_bin in range(n_bins):

        # Record the upper and lower terminals
        lower_terminals.append(
            lower_terminal
        )
        upper_terminals.append(
            lower_terminal + bin_width
        )

        # Numeric integration between lower and upper terminals
        heights.append(
            numeric_integrate_gamma(
                gamma_function,
                lower_terminals[-1],
                upper_terminals[-1],
            )
        )

        # Move to the next value
        lower_terminal = \
            upper_terminals[-1]

    # Find mid-points as the representative values
    mid_points = \
        [(lower + upper) / 2. for lower, upper in zip(lower_terminals, upper_terminals)]

    # Normalise using the average of the integration heights
    normalised_heights = [
        i_height / sum(heights) for i_height in heights
    ]

    # Return everything just in case
    return lower_terminals, upper_terminals, mid_points, normalised_heights, bin_width


def produce_gomes_exfig1(coeffs: list, add_hist=False, n_bins=3, x_values=50, plot_upper_limit=3.):
    """
    Produce figure equivalent to Extended Data Fig 1 of Aguas et al pre-print
    :return:
    """

    # Prelims
    x_values = np.linspace(0.1, plot_upper_limit, x_values)  # Can't go to zero under some coeffs
    gamma_plot = plt.figure()
    axis = gamma_plot.add_subplot(111)

    # For each requested coefficient of variation
    for coeff in coeffs:
        gamma_func = get_gamma(coeff)

        # Line graph of the raw function
        y_values = [gamma_func(i) for i in x_values]
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


# produce_gomes_exfig1(coeffs=[0.5], x_values=100, n_bins=10, add_hist=True).savefig("gomes_exfig1.jpg")
#
# lower_terminals, upper_terminals, mid_points, normalised_heights, bin_width = get_gamma_data(3., 10, 0.5)
# print(f"lower terminals: {lower_terminals}")
# print(f"upper terminals: {upper_terminals}")
# print(f"normalised heights: {normalised_heights}")
# print(f"bin width: {bin_width}")
# print(sum(normalised_heights))
