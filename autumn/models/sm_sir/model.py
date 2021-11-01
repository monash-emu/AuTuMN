from summer import CompartmentalModel
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path

from .parameters import Parameters
from datetime import date, datetime

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """

    params = Parameters(**params)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=["susceptible", "infectious", "recovered"],
        infectious_compartments=["infectious"],
        timestep=params.time.step,
        ref_date=BASE_DATE
    )

    # Check build_options
    # This will be automatically populated by calibration.py if we are running a calibration,
    # but can be manually set if so desired
    if build_options:
        validate = build_options.get("enable_validation")
        if validate is not None:
            model.set_validation_enabled(validate)
        idx_cache = build_options.get("derived_outputs_idx_cache")
        if idx_cache:
            model._set_derived_outputs_index_cache(idx_cache)

    """
    Create the total population.
    """
    init_pop = {
        "infectious": params.infectious_seed
    }

    # Get country population by age-group
    country = params.country
    total_pops = inputs.get_population_by_agegroup(["0", "50"], country.iso3, region=None, year=2020)

    # Assign the remainder starting population to the S compartment
    init_pop["susceptible"] = sum(total_pops) - sum(init_pop.values())
    model.set_initial_population(init_pop)

    """
    Add intercompartmental flows.
    """

    contact_rate = params.contact_rate

    # Infection
    model.add_infection_frequency_flow(
        name="infection",
        contact_rate=contact_rate,
        source="susceptible",
        dest="infectious",
    )

    # Recovery
    recovery_rate = 1. / params.infection_duration
    model.add_transition_flow(
        name="recovery",
        fractional_rate=recovery_rate,
        source="infectious",
        dest="recovered",
    )

    """
    Set up derived output functions
    """

    return model
