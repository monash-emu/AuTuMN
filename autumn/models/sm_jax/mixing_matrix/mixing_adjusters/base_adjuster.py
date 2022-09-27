from abc import ABC, abstractmethod

import numpy as np


class BaseMixingAdjuster(ABC):
    """
    A class used to build a time-varying mixing matrix adjustment.
    This could be adjusted according to the age or the location structure of the model's contact matrices.

    """

    @abstractmethod
    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Returns a new mixing matrix, modified to adjust for dynamic mixing
        changes for a given point in time.
        """

        pass
