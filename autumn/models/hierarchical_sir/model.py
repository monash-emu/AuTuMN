from typing import List, Tuple
from matplotlib.pyplot import summer
import pandas as pd

from summer import CompartmentalModel, Stratification

from autumn.tools.project import Params, build_rel_path

from autumn.tools.utils.summer import FunctionWrapper

from .outputs import HierarchicalSirOutputsBuilder
from .parameters import Parameters, Time
from summer.compute import ComputedValueProcessor
from .constants import BASE_COMPARTMENTS, Compartment, FlowName

from autumn.settings.constants import COVID_BASE_DATETIME

# Base date used to calculate mixing matrix times
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)

def build_model(
        params: dict,
        build_options: dict = None
) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.

    Args:
        params: The validated user-requested parameters
        build_options:

    Returns:
        The "SM-SIR" model, currently being used only for COVID-19

    """

    # Get the parameters and extract some of the more used ones to have simpler names
    params = Parameters(**params)
    time_params = params.time

    # Create the model object
    model = CompartmentalModel(
        times=(time_params.start, time_params.end),
        compartments=BASE_COMPARTMENTS,
        infectious_compartments=[Compartment.INFECTIOUS],
        timestep=time_params.step,
        ref_date=COVID_BASE_DATETIME,
    )

    """
    Check build options
    """
    # This will be automatically populated by calibration.py if we are running a calibration, but can be manually set
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    # Assign the population to compartments
    susceptible = params.total_pop - params.infectious_seed
    init_pop = {
        Compartment.INFECTIOUS: params.infectious_seed,
        Compartment.SUSCEPTIBLE: susceptible,
    }
    model.set_initial_population(init_pop)
   
    # Add the process of infecting the susceptibles for the first time
    model.add_infection_frequency_flow(
        name=FlowName.INFECTION,
        contact_rate=params.beta,
        source=Compartment.SUSCEPTIBLE,
        dest=Compartment.INFECTIOUS,
    )

    # Stratify by location
    locations = list(params.location_split.keys())
    geo_stratification = Stratification("geography", locations, BASE_COMPARTMENTS)
    geo_stratification.set_population_split(params.location_split)
    model.stratify_with(geo_stratification)

    # Add recovery flow
    model.add_transition_flow(
        name=FlowName.RECOVERY, 
        fractional_rate=params.gamma,
        source=Compartment.INFECTIOUS,
        dest=Compartment.RECOVERED
        )

    # Outputs
    outputs_builder = HierarchicalSirOutputsBuilder(model, BASE_COMPARTMENTS)
    outputs_builder.request_incidence(
    )

    return model
