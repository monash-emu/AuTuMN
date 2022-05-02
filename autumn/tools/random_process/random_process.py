from math import ceil
from math import log, sqrt, pi

from autumn.tools.curve import scale_up_function


class RandomProcess:
    """
    Defines a random process using an auto-regressive model.
    If n is the order, the process is defined as follows:
      W_k = coeff_1 * W_{k-1} + ... + coeff_n * W{k-n}  + epsilon_k ,
    where epsilon_k ~ Normal(0, sigma), and coeff_i are real constants.
    Initial conditions:
    W_0 = epsilon_0
    W_1 = coeff_1 * W_0 + epsilon_1
    """

    def __init__(self, order: int, period: int, start_time: float, end_time: float):

        assert order >= 1, "order must be >= 1"
        assert period >= 1, "period must be >= 1"
        assert end_time >= start_time + period, "at least one period must span between start_time and end_time"

        self.order = order
        self.period = period
        self.start_time = start_time
        self.end_time = end_time

        # initialise AR model parameters (coefficients and noise sd)
        self.coefficients = [1. / order] * order
        self.noise_sd = 1.

        # initialise update times and values
        n_updates = ceil((end_time - start_time) / period)
        self.update_times = [start_time + i * period for i in range(n_updates)]
        self.values = [0.] * n_updates

    def update_config_from_params(self, rp_params):
        if rp_params.values:
            msg = f"Incorrect number of specified random process values. Expected {len(self.values)}, found {len(rp_params.values)}."
            assert len(self.values) == len(rp_params.values), msg
            self.values = rp_params.values
        if rp_params.noise_sd:
            self.noise_sd = rp_params.noise_sd
        if rp_params.coefficients:
            msg = f"Incorrect number of specified coefficients. Expected {len(self.coefficients)}, found {len(rp_params.coefficients)}."
            assert len(self.coefficients) == len(rp_params.coefficients), msg
            self.coefficients = rp_params.coefficients

    def create_random_process_function(self, transform_func=None):
        """
        Create a time-variant function to be used in the main model code where the random process is implemented.
        :param transform_func: function used to transform the R interval into the desired interval
        :return: a time-variant function
        """
        if transform_func is None:
            values = self.values
        else:
            values = [transform_func(v) for v in self.values]

        sc_func = scale_up_function(self.update_times, values, method=4)

        def random_process_function(t):
            return sc_func(t)

        return random_process_function

    def evaluate_rp_loglikelihood(self):
        """
        Evaluate the log-likelihood of the process's values, given the AR coefficients and a value of noise standard deviation
        :return: the loglikelihood (float)
        """
        # calculate the centre of the normal distribution followed by each W_t
        normal_means = [
            sum([self.coefficients[k] * self.values[i - k - 1] for k in range(self.order) if i > k])
            for i in range(len(self.values))
        ]

        # calculate the distance between each W_t and the associated normal distribution's centre
        sum_of_squares = sum([(x - mu)**2 for (x, mu) in zip(self.values, normal_means)])

        # calculate the joint log-likelihood
        log_likelihood = - len(self.values) * log(self.noise_sd * sqrt(2. * pi)) - sum_of_squares / (2. * self.noise_sd**2)

        return log_likelihood

