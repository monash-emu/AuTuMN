from abc import ABC, abstractmethod

import numpy as np


class BaseMixingAdjustment(ABC):
    """
    A class used to build a time varying mixing matrix adjustment
    """

    @abstractmethod
    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Returns a new mixing matrix, modified to adjust for dynamic mixing
        changes for a given point in time.
        """
        pass
