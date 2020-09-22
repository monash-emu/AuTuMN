import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt


def get_gamma(x: float, coeff_var: float):
    """
    Use function described in caption to Extended Data Fig 1 of Aguas et al pre-print to produce gamma distribution
    from coefficient of variation and independent variable.

    :param x:
    Independent variable
    :param coeff_var:
    Independent variable
    :return:
    """

    recip_cv_2 = coeff_var ** -2.

    return \
        x ** (recip_cv_2 - 1.) * \
        np.exp(-x * recip_cv_2) / \
        special.gamma(recip_cv_2) / \
        coeff_var ** (2. * recip_cv_2)


def produce_gomes_exfig1():
    """
    Produce figure equivalent to Extended Data Fig 1 of Aguas et al pre-print
    :return:
    """

    gamma_plot = plt.figure()
    axis = gamma_plot.add_subplot(1, 1, 1)
    for coeff in [0.5, 1., 2.]:
        x_values = np.linspace(0.1, 3., 50)
        y_values = [get_gamma(i, coeff) for i in x_values]
        axis.plot(x_values, y_values)
    return gamma_plot
