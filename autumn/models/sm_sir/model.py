from summer import CompartmentalModel
from autumn.tools import inputs
from autumn.tools.project import Params, build_rel_path
from autumn.tools.random_process import RandomProcess
from math import exp

from.outputs import SmSirOutputsBuilder
from .parameters import Parameters
from datetime import date, datetime

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
base_params = Params(build_rel_path("params.yml"), validator=lambda d: Parameters(**d), validate=False)

COMPARTMENTS = ["susceptible", "infectious", "recovered"]


def build_model(params: dict, build_options: dict = None) -> CompartmentalModel:
    """
    Build the compartmental model from the provided parameters.
    """

    params = Parameters(**params)

    # Create the model object
    model = CompartmentalModel(
        times=(params.time.start, params.time.end),
        compartments=COMPARTMENTS,
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
    # transmission
    if params.activate_random_process:
        # build the random process, using default values and coefficients
        rp = RandomProcess(order=2, period=7, start_time=params.time.start, end_time=params.time.end)

        # set coefficients and values if specified in the parameters
        rp_params = params.random_process
        if rp_params.values:
            msg = f"Incorrect number of specified random process values. Expected {len(rp.values)}, found {len(rp_params.values)}."
            assert len(rp.values) == len(rp_params.values), msg
            rp.values = rp_params.values
        if rp_params.noise_sd:
            rp.noise_sd = rp_params.noise_sd
        if rp_params.coefficients:
            msg = f"Incorrect number of specified coefficients. Expected {len(rp.coefficients)}, found {len(rp_params.coefficients)}."
            assert len(rp.coefficients) == len(rp_params.coefficients), msg
            rp.coefficients = rp_params.coefficients

        # FIXME: Need to check with David S. as this does not look good.
        model.random_processes = rp

        # Create function returning exp(W), where W is the random process
        contact_rate = rp.create_random_process_function(transform_func=lambda w: params.contact_rate * exp(w))
    else:
        contact_rate = params.contact_rate

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
    outputs_builder = SmSirOutputsBuilder(model, COMPARTMENTS)
    outputs_builder.request_incidence()

    return model
