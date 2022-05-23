from typing import List, Dict, Optional, Union

from scipy import stats
import numpy as np
from numba import jit

from autumn.tools.utils.summer import OutputsBuilder
from .constants import Compartment, FlowName



class HierarchicalSirOutputsBuilder(OutputsBuilder):

    def request_incidence(
            self
    ):
        """
        Calculate incident disease cases. This is associated with the transition to infectiousness if there is only one
        infectious compartment, or transition between the two if there are two.
        Note that this differs from the approach in the covid_19 model, which took entry to the first "active"
        compartment to represent the onset of symptoms, which infectiousness starting before this.

        Args:
            age_groups: The modelled age groups
            clinical_strata: The clinical strata implemented
            strain_strata: The modelled strains, or None if model is not stratified by strain
            incidence_flow: The name of the flow representing incident cases

        """

        # Unstratified
        self.model.request_output_for_flow(name="incidence", flow_name=FlowName.INFECTION)