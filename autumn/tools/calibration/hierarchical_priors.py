from typing import List, Union

class HierarchicalPrior():

    def __init__(self, param_name: str, distribution: str, hyper_parameters: List[Union[str, float]]):
        """
        Args:
            param_name: the name of the cluster-specific parameter
            distribution: name of a statistical distribution
            hyper_parameters: list of hyper-parameters informing the distribution. Hyper-parameters defined with a string are also calibrated.
        """
        self.param_name = param_name
        self.distribution = distribution
        self.hyper_parameters = hyper_parameters
